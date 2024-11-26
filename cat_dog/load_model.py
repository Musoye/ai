import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

def predict_image(filename):
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*128*128, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
    )

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])

    model.load_state_dict(torch.load("cat_dog.ph"))
    model.eval()


    image = Image.open(filename)
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)

    value1 = output[0, 0].item()
    value2 = output[0, 1].item() 

    if value1 > value2:
        return "cat"
    return "dog"