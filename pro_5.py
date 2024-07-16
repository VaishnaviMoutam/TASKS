import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# Define transformations
transform = Compose([
    Resize((28, 28)),
    ToTensor()
])

# Define data paths
training_path = "/home/vaishnavi-moutam/Downloads/final/Training1"
validation_path = "/home/vaishnavi-moutam/Downloads/final/Validation1"
testing_path = "/home/vaishnavi-moutam/Downloads/final/Testing1"

# Load training data
train_samples = []
for root, _, fnames in sorted(os.walk(training_path)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        try:
            img = Image.open(path).convert('L')
            train_samples.append((transform(img), 0))
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")
train_loader = DataLoader(train_samples, batch_size=32, shuffle=True)

# Load validation data
val_samples = []
for root, _, fnames in sorted(os.walk(validation_path)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        try:
            img = Image.open(path).convert('L')
            val_samples.append((transform(img), 0))
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")
val_loader = DataLoader(val_samples, batch_size=32, shuffle=False)

# Load testing data
test_samples = []
for root, _, fnames in sorted(os.walk(testing_path)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        try:
            img = Image.open(path).convert('L')
            test_samples.append((transform(img), 0))
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")
test_loader = DataLoader(test_samples, batch_size=32, shuffle=False)

# Define model architecture
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Define optimizers to compare
optimizers = {
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'SGD': optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001)
}

# Training and evaluation loop for each optimizer
num_epochs = 3
for optimizer_name, optimizer in optimizers.items():
    print(f"Training with {optimizer_name} optimizer:")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Calculate training accuracy
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_acc = correct / total

        # Calculate validation accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f"Testing Accuracy with {optimizer_name} optimizer: {test_acc:.4f}")
    print()
