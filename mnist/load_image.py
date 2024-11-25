import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

def predict_image(filename):
    hidden_units = [32, 16]
    image_size = (1, 28, 28)
    input_size = image_size[0] * image_size[1] * image_size[2]

    all_layers = [nn.Flatten()]
    for hidden_unit in hidden_units:
        all_layers.append(nn.Linear(input_size, hidden_unit))
        all_layers.append(nn.ReLU())
        input_size = hidden_unit
    all_layers.append(nn.Linear(hidden_units[-1], 10))

    model = nn.Sequential(*all_layers)

    model.load_state_dict(torch.load("./mnist_classifier"))
    model.eval()


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),  # Convert to tensor
        #transforms.Normalize((0.5,), (0.5,))  # Normalize the image
    ])


    image = Image.open(filename)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    output = model(image_tensor)

    # Get the predicted class
    predicted_class = output.argmax(dim=1).item()

    return predicted_class