# Tannic-NN Example: Serving a Vision Transformer (ViT)

This repository contains a minimal example of how to use [Tannic-NN](https://github.com/entropy-flux/Tannic-NN) to serve a **Vision Transformer (ViT)** model over a simple socket-based server.  

It is intended as a **baseline reference implementation**, a ground to build upon showing the essential steps of serving a model with Tannic-NN in the simplest possible way.  
The Tannic tensor library is still unoptimized, **it is not optimized and therefore runs relatively slow**. The goal is to provide a transparent starting point, not a production-ready server.

This example demonstrates:  
- Initializing model parameters (pretrained weights)  
- Running inference with a ViT model built with **Tannic-NN**  
- Sending and receiving tensors through a custom TCP server with a python client

---

## Example

The code sets up a `Server` listening on port `8080` and exposes a **Vision Transformer (B-16, ImageNet-1k)** for inference.  
Clients can connect, send input tensors (e.g., images), and receive the model’s predictions.

Key components:
- **ViT Model** (`include/vit.hpp`): Defined using Tannic-NN building blocks.  
- **Server** (`include/server.hpp`): Minimal TCP server for exchanging tensors.    
- **Client** (`evaluate.py`): Minimal python client sending a picture an printing probabilities. 


### Demo: Classifying a dog

As a test, we send the image [`data/pug_600.jpg`](data/pug_600.jpg) to the server.  

![Pug example](data/pug_600.jpg)

The model correctly identifies the dog: 

1. pug (78.32%)
2. Brabancon griffon (4.88%)
3. bull mastiff (2.50%)
4. Labrador retriever (1.38%)
5. French bulldog (0.99%)

---

## 📦 Dependencies 

- A C++23 compiler (GCC ≥ 14, Clang ≥ 16, etc.)
- CMake ≥ 3.30

[Download the weights from Hugging Face](https://huggingface.co/entropy-flux/vit-imagenet1k/tree/main). Metadata is already in the repo so only the `vit-imagenet1k-B-16.tannic` file is required. 

And place the file under `data/` folder. You can also download directly from command line:

```bash 
wget -O data/vit-imagenet1k-B-16.tannic https://huggingface.co/entropy-flux/vit-imagenet1k/resolve/main/vit-imagenet1k-B-16.tannic
``` 

You can also build the tannic weights directly. Donwload the `vit-imagenet1k-B-16.pth` file from HF, load them in the /model/vit.py python model and use the [PyTannic](https://github.com/entropy-flux/PyTannic) package to write the model as a .tannic file. 

You will need PyTorch and PyTannic (Just for the example, the server doesn't rely on python.)
PyTannic is a binding I created to easily send torch tensors to the server, you can install it with pip just as:

```bash
pip install pytannic
```

For the example you should also install the `requests` package, since it will try to download an image from internet. 

```bash
pip install requests
```

---

## 🛠️ Build Instructions
  
Clone this repo (with submodules) alongside **Tannic-NN**:

```bash
git clone --recursive https://github.com/your-username/vit-server-example.git
cd vit-server-example
mkdir build && cd build
cmake ..
make -j$(nproc) 
```
If you forgot --recursive when clone:

```bash
git submodule update --init --recursive
```

Then from the build directory run the executable:

```bash
./vit-server
```

This will setup a `Server` listening on port `8080` exposing the model. You can try it now just running:

```bash
python main.py
```

And it will download and send a picture to the server and print the results. 

⚠️ Note: The Tannic tensor library is still unoptimized, so inference may take a few minutes.
This repository is meant as a baseline reference implementation clear and minimal rather than a fast production-ready server.