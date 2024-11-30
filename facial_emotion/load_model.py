import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

def predict_image(filename):
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1*48*48, 128),
    nn.ReLU(),
    nn.Linear(128, 7)
)

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

    model.load_state_dict(torch.load("facial_emotion.ph"))
    model.eval()


    image = Image.open(filename)
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)

    arr = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    emo = []
    emo = output.tolist()[0]

    print(emo)

    max_id =  0

    for i in range(6):
        if emo[i] > emo[i + 1]:
            max_id = i
        else:
            max_id = i + 1
    
    return arr[max_id]