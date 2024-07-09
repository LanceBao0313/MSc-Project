import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from collections import defaultdict

class CIFAR10Subset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

def split_non_iid(dataset, num_clients, num_classes_per_client):
    # Get the indices of each class
    class_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)
    
    # Shuffle the indices within each class
    for indices in class_indices.values():
        random.shuffle(indices)
    
    # Allocate indices to clients
    client_indices = [[] for _ in range(num_clients)]
    
    for client_id in range(num_clients):
        selected_classes = random.sample(class_indices.keys(), num_classes_per_client)
        for cls in selected_classes:
            num_samples = len(class_indices[cls]) // num_clients
            client_indices[client_id].extend(class_indices[cls][:num_samples])
            class_indices[cls] = class_indices[cls][num_samples:]
    
    # Create a subset dataset for each client
    subsets = [Subset(dataset, indices) for indices in client_indices]
    return subsets

def get_federated_dataloaders(root, num_clients, num_classes_per_client, batch_size=32, transform=None):
    # Transformations
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # Load CIFAR-10 dataset
    cifar10_train = CIFAR10(root=root, train=True, download=False, transform=transform)
    
    # Split into non-IID subsets
    subsets = split_non_iid(cifar10_train, num_clients, num_classes_per_client)
    
    # Create DataLoaders
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
    
    return dataloaders
