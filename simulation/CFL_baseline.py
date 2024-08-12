import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset import get_nonIID_dataloader
from configuration import DATA_PATH, DEVICE, BATCH_SIZE
from model import load_checkpoint, reset_classifier_weights, save_checkpoint
from torchvision.datasets import CIFAR10
from eval import evaluate_model

def local_training(model, dataloader, criterion, optimizer, epochs=1, device='cuda'):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()

def federated_learning(global_model, client_dataloaders, num_rounds=1, num_parallel_clients=1, epochs=4, device='cuda'):
    global_model.to(device)
    global_params = [param.data.clone() for param in global_model.parameters()]

    for round_num in range(num_rounds):
        with ThreadPoolExecutor(max_workers=num_parallel_clients) as executor:
            futures = []
            for dataloader in client_dataloaders:
                local_model = load_checkpoint("./checkpoints/baseline_checkpoint_2.pth")
                reset_classifier_weights(global_model)
                local_model.to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.Adam(local_model.classifier.parameters(), lr=0.005)
                criterion = nn.CrossEntropyLoss()

                future = executor.submit(local_training, local_model, dataloader, criterion, optimizer, epochs, device)
                futures.append(future)

            for future in tqdm(as_completed(futures), desc=f"Round {round_num+1}/{num_rounds}", total=len(futures)):
                future.result()

            # Aggregate updates
            new_params = [torch.zeros_like(param) for param in global_model.classifier.parameters()]
            for dataloader in client_dataloaders:
                local_model = load_checkpoint("./checkpoints/baseline_checkpoint_2.pth")
                reset_classifier_weights(global_model)
                local_model.to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                for new_param, local_param in zip(new_params, local_model.classifier.parameters()):
                    new_param.data.add_(local_param.data / len(client_dataloaders))

            for global_param, new_param in zip(global_model.parameters(), new_params):
                global_param.data.copy_(new_param.data)

    return global_model

if  __name__ == "__main__":
    # Load CIFAR-10 dataset
    client_dataloaders = get_nonIID_dataloader(DATA_PATH, train=True)

    # Initialize the global model
    global_model = load_checkpoint("./checkpoints/baseline_checkpoint_2.pth")
    reset_classifier_weights(global_model)
    global_model.to(DEVICE)

    # Train the model using federated learning
    trained_model = federated_learning(global_model, client_dataloaders, num_rounds=1, num_parallel_clients=1, epochs=4)

    # Save the trained model
    # torch.save(trained_model.state_dict(), './checkpoints/CFL_baseline.pth')
    save_checkpoint(trained_model, optim.Adam(trained_model.parameters(), lr=0.005), './checkpoints/CFL_baseline.pth')
    model = load_checkpoint("./checkpoints/CFL_baseline.pth")
    model.to(DEVICE)
    # Evaluate the trained model
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    test_dataset = CIFAR10(root= DATA_PATH, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    accuracy, f1 = evaluate_model(trained_model, test_loader, device=DEVICE)
    print(f'Accuracy of the trained model: {accuracy:.4f}, F1 score: {f1:.4f}')




