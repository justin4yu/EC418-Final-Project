import os
import torch
import torch.optim as optim
from planner import Planner, save_model  # Your Planner model from planner.py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the transformation (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((96, 128)),  # Resize images to fit the model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Custom Dataset Class (no need to modify utils.py)
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # List all image files in the data directory (assuming .png, .jpg, .jpeg)
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Load labels (assuming labels are in 'labels.txt')
        self.labels = self.load_labels()

    # def load_labels(self):    
    #     label_file = os.path.join(self.data_dir, "labels.txt")
    #     labels = []
    #     with open(label_file, 'r') as f:
    #         for line in f:
    #             labels.append([float(x) for x in line.strip().split()])  # Assuming labels are x, y coordinates
    #     return labels

    def load_labels(self):
        labels = []
        for img_file in self.image_files:
            label_file = img_file.replace('.png', '.csv')  # Adjust based on your dataset
            labels.append(np.loadtxt(label_file, dtype=np.float32, delimiter=','))
        return labels


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Open image in RGB mode

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Initialize Dataset and DataLoader
dataset = CustomDataset('C:/Users/madak/OneDrive/Desktop/EC418/drive_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize Model, Optimizer, and Loss Function
planner = Planner()
optimizer = optim.Adam(planner.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()  # Assuming aim points are given as labels (x, y coordinates)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()  # Reset gradients
        outputs = planner(images)  # Forward pass through the Planner model
        loss = loss_fn(outputs, labels)  # Calculate loss (aim point regression)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

# Save the trained model
save_model(planner)
print("Model saved as 'planner.th'")
