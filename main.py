import requests
import io
import os
import torch
from torch.nn.functional import softmax
from PIL import Image
from torchvision import transforms
from pytannic.client import Client
from pytannic.torch.tensors import serialize, deserialize
 
URL = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
filename = "data/pug_600.jpg"

if os.path.exists(filename):
    print(f"📂 Found cached image: {filename}")
    image = Image.open(filename).convert("RGB")
else:
    print("🌐 Downloading image...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL, headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Request failed: {response.status_code}")

    with open(filename, "wb") as file:
        file.write(response.content)
    print(f"✅ Image saved as {filename}")
    image = Image.open(io.BytesIO(response.content)).convert("RGB")


transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input = transform(image)

with Client("0.0.0.0", 8080) as client:
    print("May take a while since kernels are not optimized...")
    client.send(serialize(input))
    response = client.receive()
    outputs = deserialize(response) 
 
probabilities = softmax(outputs[0], dim=0)
top5_probabilities, top5_indices = torch.topk(probabilities, 5)

URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
filename = "data/imagenet-classes.txt"

if os.path.exists(filename):
    print(f"📂 Found cached labels: {filename}")
    with open(filename, "r") as file:
        classes = file.read().splitlines()
else:
    print("🌐 Downloading ImageNet labels...")
    classes = requests.get(URL).text.splitlines()
    with open(filename, "w") as f:
        f.write("\n".join(classes))
    print(f"✅ Labels saved as {filename}")
 
print("\nTop-5 Predictions:")
for idx, (probability, index) in enumerate(zip(top5_probabilities, top5_indices)):
    print(f"{idx+1}: {classes[index]} ({probability.item()*100:.2f}%)")
