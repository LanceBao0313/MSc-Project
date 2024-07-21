from model import load_checkpoint, train, replace_fc_layer, save_checkpoint, reset_classifier_weights
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import CIFAR10, get_CIFAR10_dataloader, get_nonIID_dataloader
from configuration import DATA_PATH, DEVICE, BATCH_SIZE, NUM_OF_CLASSES, RANDOM_SEED
from mcunet.mcunet.model_zoo import build_model
from eval import evaluate_model


torch.manual_seed(RANDOM_SEED)
NUM_EPOCHS = 5
checkpoint_path = f"./checkpoints/baseline_checkpoint_2.pth"

# Load training data
# dataloader = get_CIFAR10_dataloader(DATA_PATH)
# # Load test data
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
test_dataset = CIFAR10(root= DATA_PATH, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

dataloaders = get_nonIID_dataloader(DATA_PATH, train=True)
# print(len(dataloaders))
dataloader = dataloaders[0]
# Load the model
# model, resolution, description = build_model(net_id="mcunet-in3", pretrained=True)
# model = replace_fc_layer(model, NUM_OF_CLASSES)
model = load_checkpoint(checkpoint_path)
reset_classifier_weights(model)
model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.005)

# Train the model
###  counter that limits the number of datapoints
### device -> DEVICE
### running_loss -> epoch_loss

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    counter = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
# save_checkpoint(model, optimizer, "./checkpoints/baseline_checkpoint.pth")

# Evaluate the model
accuracy, f1 = evaluate_model(model, test_loader, DEVICE)
print(f"Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")

