#include <tannic.hpp>
#include <tannic/functions.hpp>
#include <tannic/reductions.hpp>
#include <tannic/serialization.hpp>
#include <tannic-nn.hpp>
#include <tannic-nn/parameters.hpp>
#include <tannic-nn/functional.hpp>
#include <tannic-nn/convolutional.hpp>
#include "include/server.hpp" 
#include "include/vit.hpp"

using namespace tannic;

// Note: This has a lot of nasty serving code that will be abstracted in
// a serving framework. This is just for the sake of the example.

int main() {   
    nn::Parameters parameters; parameters.initialize("../data/vit-imagenet1k-B-16");  
    ViT model(float32, 16, 3, 384, 768, 3072, 12, 12, 1000); model.initialize(parameters);

    Server server(8080);
    while (true) {
        Socket socket = server.accept();  

        try {
            while (true) {
                Header header{};
                if (!server.read(socket, &header, sizeof(Header))) {
                    std::cout << "Client disconnected.\n";
                    break; 
                }

                if (header.magic != MAGIC) {
                    std::cerr << "Invalid magic! Closing connection.\n";
                    break;
                }

                Metadata<Tensor> metadata{};  
                if (!server.read(socket, &metadata, sizeof(Metadata<Tensor>))) {
                    std::cout << "Client disconnected.\n";
                    break;
                }

                Shape shape; 
                size_t size;
                for (uint8_t dimension = 0; dimension < metadata.rank; dimension++) {
                    if (!server.read(socket, &size, sizeof(size_t))) {
                        std::cout << "Client disconnected.\n";
                        break;
                    }
                    shape.expand(size);
                }

                std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(metadata.nbytes);
                if (!server.read(socket, buffer->address(), metadata.nbytes)) {
                    std::cout << "Client disconnected.\n";
                    break;
                }

                Tensor input(dtypeof(metadata.dcode), shape, 0, buffer);  
                Tensor output = repack(model(input)); 

                header = headerof(output);
                metadata = metadataof(output);

                server.write(socket, &header, sizeof(Header));
                server.write(socket, &metadata, sizeof(Metadata<Tensor>)); 
                server.write(socket, output.shape().address(), output.shape().rank() * sizeof(size_t));
                server.write(socket, output.bytes(), output.nbytes());
            }

        } catch (const std::exception& e) {
            std::cerr << "Unexpected client error: " << e.what() << "\n";
        }
    }    
} 