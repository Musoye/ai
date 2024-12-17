import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

def predict_image(filename):
    model = nn.Sequential()

    #featurizer
    n_ch = 1
    model.add_module('c1', nn.Conv2d(
        in_channels=1, out_channels=n_ch, kernel_size=3
    ))
    model.add_module('relu1', nn.ReLU())
    model.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
    model.add_module('flatten', nn.Flatten())

    #classfication
    model.add_module('fc1', nn.Linear(n_ch*13*13, 70))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(70, 35))
    model.add_module('relu3', nn.ReLU())
    model.add_module('fc4', nn.Linear(35, 12))
    model.add_module('relu4', nn.ReLU())
    model.add_module('fc2', nn.Linear(12, 7))

    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

    model.load_state_dict(torch.load("facial_emotion.ph"))
    model.eval()


    image = Image.open(filename)
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)

    predict =  int((torch.argmax(output, dim=1)).float())

    arr = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    return arr[predict]