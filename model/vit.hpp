#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>
#include <memory>
#include <tannic.hpp> 
#include <tannic/reductions.hpp>
#include <tannic/transformations.hpp>
#include <tannic/serialization.hpp>
#include <tannic-nn.hpp> 
#include <tannic-nn/functional.hpp> 
#include <tannic-nn/convolutional.hpp>
#include "server.hpp" 

using namespace tannic;  

struct LayerNorm : public nn::Module {
    nn::Parameter weight;
    nn::Parameter bias;
    float epsilon;
    Shape shape; 
 
    constexpr LayerNorm(type dtype, Shape shape, float epsilon = 1e-6f)
    :   weight(dtype, shape)
    ,   bias(dtype, shape)
    ,   epsilon(epsilon)
    ,   shape(shape) { 
    }

    constexpr LayerNorm(type dtype, size_t dimension, float epsilon = 1e-6f)
    :   LayerNorm(dtype, Shape(dimension), epsilon) { 
    } 

    Tensor forward(Tensor const& input) const { 
        auto mu = mean(input, -shape.rank(), true); 
        auto centered = input - mu;           
        auto squared = centered * centered;  
        auto variance = mean(squared, -shape.rank(), true);  
        auto normalized = centered * rsqrt(variance, epsilon);    
        return normalized * weight + bias; 
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        weight.initialize(name + ".weight", parameters);
        bias.initialize(name + ".bias", parameters);
    }
}; 

Tensor scaled_dot_attention(Tensor q, Tensor k, Tensor v) { 
    auto scale   = 1 / std::sqrt(k.size(-1));    
    Tensor product = matmul(q, k.transpose(-2, -1)) * scale;  
    Tensor scores   = nn::softmax(product, -1); 
    return matmul(scores, v);
} 

struct Attention : nn::Module {
    nn::Linear q_projection;
    nn::Linear k_projection;
    nn::Linear v_projection;

    size_t number_of_heads;
    size_t model_dimension;
    size_t heads_dimension;

    constexpr Attention(type dtype, size_t model_dimension, size_t number_of_heads)
    :   model_dimension(model_dimension)
    ,   number_of_heads(number_of_heads)
    ,   heads_dimension(model_dimension / number_of_heads) 
    ,   q_projection(dtype, model_dimension, model_dimension)
    ,   k_projection(dtype, model_dimension, model_dimension)
    ,   v_projection(dtype, model_dimension, model_dimension) {
        if (model_dimension % number_of_heads != 0)
            throw Exception("Model dimension must be divisible by number of heads");
    }


    void initialize(std::string const& name, nn::Parameters& parameters) const {
        q_projection.initialize(name + ".q_projection", parameters);
        k_projection.initialize(name + ".k_projection", parameters);
        v_projection.initialize(name + ".v_projection", parameters);
    }

    Tensor split(Tensor sequence) const {  
        sequence = sequence.view(sequence.size(0), sequence.size(1), number_of_heads, model_dimension / number_of_heads);
        return sequence.transpose(1, 2);
    }

    Tensor merge(Tensor sequence) const {
        sequence = sequence.transpose(1, 2); 
        return reshape(sequence, sequence.size(0) , sequence.size(1), heads_dimension * number_of_heads);
    }

    Tensor forward(Tensor sequence) const {
        auto q = split(q_projection(sequence));
        auto k = split(k_projection(sequence));
        auto v = split(v_projection(sequence));  
        sequence = scaled_dot_attention(q, k, v);  
        return merge(sequence);
    } 
};  


struct FFN : nn::Module {
    nn::Linear input_layer;
    nn::Linear output_layer;

    constexpr FFN(type dtype, size_t model_dimension, size_t hidden_dimension) 
    :   input_layer(dtype, model_dimension, hidden_dimension)
    ,   output_layer(dtype, hidden_dimension, model_dimension)
    {}

    Tensor forward(Tensor features) const {
        features = nn::gelu(input_layer(features));
        return output_layer(features);
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        input_layer.initialize(name + ".input_layer", parameters);
        output_layer.initialize(name + ".output_layer", parameters); 
    }
}; 
 

struct Encoder : nn::Module {
    Attention attention;
    LayerNorm attention_norm;
    nn::Linear projection;
    FFN ffn;
    LayerNorm ffn_norm;

    constexpr Encoder(type dtype, size_t model_dimension, size_t number_of_heads, size_t ffn_hidden_dimension)
    :   attention(dtype, model_dimension, number_of_heads)
    ,   attention_norm(dtype, model_dimension, 1e-6)
    ,   projection(dtype, model_dimension, model_dimension)
    ,   ffn(dtype, model_dimension, ffn_hidden_dimension)
    ,   ffn_norm(dtype, model_dimension, 1e-6)
    {} 
 
    Tensor forward(Tensor sequence) const { 
        sequence = projection(attention(attention_norm(sequence))) + sequence;
        return ffn(ffn_norm(sequence)) + sequence;
    }
 
    void initialize(std::string const& name, nn::Parameters& parameters) const {
        attention.initialize(name + ".attention", parameters);
        attention_norm.initialize(name + ".attention_norm", parameters);
        projection.initialize(name + ".projection", parameters);
        ffn.initialize(name + ".ffn", parameters);
        ffn_norm.initialize(name + ".ffn_norm", parameters);
    }

};  

struct Transformer : nn::Module {
    size_t number_of_layers;
    nn::List<Encoder> encoders;

    Transformer(type dtype, size_t number_of_layers, size_t model_dimension, size_t number_of_heads, size_t ffn_hidden_dimension) {
        for (size_t layer_number = 0; layer_number < number_of_layers; layer_number++) {
            encoders.add(Encoder(dtype, model_dimension, number_of_heads, ffn_hidden_dimension));
        }
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        size_t layer_number = 0;
        for(auto const& encoder: encoders) {
            encoder.initialize(name + ".encoders." + std::to_string(layer_number), parameters);
            layer_number++;
        }
    }

    Tensor forward(Tensor sequence) const {    
        for(auto const& encoder: encoders) { 
            std::cout << "Encoding..." << std::endl;
            sequence = encoder(sequence);  
        }  
        return sequence;
    }
};
 
 

struct Patcher : nn::Module {
    size_t image_size;
    size_t patch_size;
    nn::Convolutional2D projection;
    
    constexpr Patcher(type dtype, size_t image_size, size_t patch_size, size_t input_channels, size_t model_dimension)
    :   image_size(image_size)
    ,   patch_size(patch_size)
    ,   projection(dtype, input_channels, model_dimension, {patch_size, patch_size}, {patch_size, patch_size}) 
    {
        if (image_size % patch_size != 0)
            throw Exception("Image dimensions must be divisible by patch size");
    }
 
   
    void initialize(std::string const& name, nn::Parameters& parameters) const { 
        projection.initialize(name + ".projection", parameters);
    }

    Tensor forward(Tensor image) const {  
        auto features = projection(unsqueeze(image, 0));          
        return transpose(flatten(features, 2), 1, 2);
    }
};
   
struct CLSToken : nn::Module {
    nn::Parameter weight;
    
    constexpr CLSToken(type dtype, size_t model_dimension)
    : weight(dtype, Shape(1, 1, model_dimension)) {}
 
    Tensor forward(Tensor sequence) const {
        size_t batch_size = sequence.size(0);
        auto cls = expand(weight, batch_size, -1, -1); 
        return concatenate(cls, sequence, 1);
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const { 
        weight.initialize(name + ".weight", parameters);
    }
}; 
   
struct ViT : nn::Module { 
    Patcher patcher;
    nn::Parameter positions;
    CLSToken cls_token;
    Transformer transformer;
    LayerNorm norm;
    nn::Linear head; 
    ViT(
        type dtype, 
        size_t patch_size, 
        size_t input_channels,
        size_t image_size,
        size_t model_dimension,
        size_t ffn_hidden_dimension,
        size_t number_of_layers,
        size_t number_of_heads, 
        size_t number_of_classes 
    ) 
    :   patcher(dtype, image_size, patch_size, input_channels, model_dimension)    
    ,   cls_token(dtype, model_dimension)
    ,   positions(dtype, Shape(1, 1 + (patcher.image_size / patcher.patch_size) *  (patcher.image_size / patcher.patch_size), model_dimension))
    ,   transformer(dtype, number_of_layers, model_dimension, number_of_heads, ffn_hidden_dimension)
    ,   norm(dtype, model_dimension, 1e-6)
    ,   head(dtype, model_dimension, number_of_classes) {}
  

    Tensor forward(Tensor sequence) const {      
        sequence = patcher(sequence);        
        sequence = cls_token(sequence) + positions;    
        sequence = transformer(sequence);      
        sequence = norm(sequence);    
        return head(sequence[{0,-1}][0]);   
    }

    void initialize(nn::Parameters& parameters) const {
        patcher.initialize("patcher", parameters);
        positions.initialize("positions", parameters);
        cls_token.initialize("cls_token", parameters);
        transformer.initialize("transformer", parameters);
        norm.initialize("norm", parameters);
        head.initialize("head", parameters);
    }
}; 

