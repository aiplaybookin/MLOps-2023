import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, LogSoftmax, Flatten
import os
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = Sequential(
            Conv2d(1, 32, 3, 1),
            ReLU(),
            Conv2d(32, 64, 3, 1),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(9216, 128),
            ReLU(),
            Linear(128, 10),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)


# Define the path to the pre-trained model
model_path = './model/mnist_cnn.pt'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load the pre-trained model
model = Net().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transform to apply to the input images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
dataset = MNIST(root='./data', train=False, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Create the results directory if it doesn't exist
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Perform inference on 5 random images
for i, (image, _) in enumerate(dataloader, 1):
    if i > 5:
        break

    # Forward pass through the model
    output = model(image)

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)
    predicted_number = predicted.item()

    # Save the image with the predicted number as the file name
    #image_path = os.path.join(results_dir, f'predicted_{predicted_number}.png')
    
    # Convert the image tensor to a NumPy array
    image_np = image.squeeze().numpy()

    # Create a figure and plot the image
    fig, ax = plt.subplots()
    ax.imshow(image_np, cmap='gray')
    ax.axis('off')

    # Save the figure with the predicted number as the file name
    image_path = os.path.join(results_dir, f'predicted_{predicted_number}.png')
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved result for image {i}: predicted number = {predicted_number}')
