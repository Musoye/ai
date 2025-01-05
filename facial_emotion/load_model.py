import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class EmotionClassification(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassification, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Classification
        self.fc1 = nn.Linear(128 * 6 * 6, 256) 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Featurizer
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Classifier
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 

        return x


def predict_image(filename):
    model = EmotionClassification(num_classes=7)

    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
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