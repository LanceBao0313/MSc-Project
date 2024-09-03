import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from collections import defaultdict
from configuration import BATCH_SIZE, NUMBER_OF_DEVICES, NUM_OF_CLASSES, NON_IID_ALPHA, RANDOM_SEED, DATASET
from typing import List, Tuple
from collections import Counter
from scipy.stats import wasserstein_distance

# class CIFAR10Subset(Dataset):
#     def __init__(self, data, targets, transform=None):
#         self.data = data
#         self.targets = targets
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img, target = self.data[idx], self.targets[idx]
#         if self.transform:
#             img = self.transform(img)
#         return img, target
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# def split_non_iid(dataset, num_clients, num_classes_per_client):
#     # Get the indices of each class
#     class_indices = defaultdict(list)
#     for idx, target in enumerate(dataset.targets):
#         class_indices[target].append(idx)
    
#     # Shuffle the indices within each class
#     for indices in class_indices.values():
#         random.shuffle(indices)
    
#     # Allocate indices to clients
#     client_indices = [[] for _ in range(num_clients)]
    
#     for client_id in range(num_clients):
#         selected_classes = random.sample(class_indices.keys(), num_classes_per_client)
#         for cls in selected_classes:
#             num_samples = len(class_indices[cls]) // num_clients
#             client_indices[client_id].extend(class_indices[cls][:num_samples])
#             class_indices[cls] = class_indices[cls][num_samples:]
    
#     # Create a subset dataset for each client
#     subsets = [Subset(dataset, indices) for indices in client_indices]
#     return subsets

# def get_federated_dataloaders(root, num_clients, num_classes_per_client, batch_size=64, transform=None):
#     # Transformations
#     if transform is None:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])

#     # Load CIFAR-10 dataset
#     cifar10_train = CIFAR10(root=root, train=True, download=False, transform=transform)
    
#     # Split into non-IID subsets
#     subsets = split_non_iid(cifar10_train, num_clients, num_classes_per_client)
    
#     # Create DataLoaders
#     dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
    
#     return dataloaders

def get_dataloader(root, train=True):
    if DATASET == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load the CIFAR-10 dataset
        dataset = CIFAR10(root=root, train=train, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        dataset = MNIST(root=root, train=train, download=False, transform=transform)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return dataloader

def get_IID_dataloader(root, train=True):
    if DATASET == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load the CIFAR-10 dataset
        dataset = CIFAR10(root=root, train=train, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        dataset = MNIST(root=root, train=train, download=False, transform=transform)
    
    # Get the total number of samples in the dataset
    num_samples = len(dataset)
    
    # Calculate the size of each subset
    subset_size = num_samples // NUMBER_OF_DEVICES
    
    # Generate random indices to split the dataset
    indices = torch.randperm(num_samples).tolist()
    
    # Split the indices into subsets
    subset_indices = [indices[i * subset_size:(i + 1) * subset_size] for i in range(NUMBER_OF_DEVICES)]
    
    # Handle any remaining samples (due to integer division)
    if len(indices) > subset_size * NUMBER_OF_DEVICES:
        subset_indices[-1].extend(indices[NUMBER_OF_DEVICES * subset_size:])
    
    # Create DataLoaders for each subset
    dataloaders = []
    for subset_idx in subset_indices:
        subset = Subset(dataset, subset_idx)
        dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        dataloaders.append(dataloader)
    
    return dataloaders

def get_nonIID_dataloader(root, train=True):
    if DATASET == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load the CIFAR-10 dataset
        dataset = CIFAR10(root=root, train=train, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        dataset = MNIST(root=root, train=train, download=False, transform=transform)
    

    prng = np.random.default_rng(RANDOM_SEED)
    tmp_t = np.array(dataset.targets)
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    
    idx_clients = [[] for _ in range(NUMBER_OF_DEVICES)]

    # Distribute data according to Dirichlet distribution
    for k in range(num_classes):
        idx_k = np.where(tmp_t == k)[0]
        prng.shuffle(idx_k)
        
        # Generate Dirichlet proportions for current class
        proportions = prng.dirichlet(np.repeat(NON_IID_ALPHA, NUMBER_OF_DEVICES))
        
        # Normalize proportions to ensure they sum to 1 and then scale by the number of samples
        proportions = np.array([int(p * len(idx_k)) for p in proportions])
        
        # Ensure the total number of samples is correctly partitioned
        while proportions.sum() < len(idx_k):
            proportions[prng.choice(len(proportions))] += 1
        while proportions.sum() > len(idx_k):
            proportions[prng.choice(len(proportions))] -= 1
        
        # Split indices based on calculated proportions
        idx_k_split = np.split(idx_k, np.cumsum(proportions)[:-1])
        idx_clients = [idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)]

    trainsets_per_client = [Subset(dataset, idxs) for idxs in idx_clients]
   
    # Create data loaders for each client
    client_dataloaders = [
        DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        for dataset in trainsets_per_client
    ]

    return client_dataloaders

def calculate_label_distribution(dataloader):
    # Count the occurrences of each label in the dataloader
    label_counts = Counter()
    for _, label in dataloader:
        label_counts.update(label.numpy().tolist())
    
    # Convert counts to distribution (as a probability distribution)
    total_labels = sum(label_counts.values())
    label_distribution = {label: count / total_labels for label, count in label_counts.items()}
    
    return label_distribution

def calculate_emd(distribution):

    # Assuming a uniform global distribution for CIFAR-10
    global_distribution = np.array([0.1] * 10)  # Each class has equal probability of 0.1

    # Convert the subset distribution to an array (in the order of labels 0 to 9)
    subset_array = np.array([distribution.get(i, 0) for i in range(10)])
    
    # Calculate Earth Mover's Distance (EMD) between the subset and global distributions
    emd = wasserstein_distance(subset_array, global_distribution)
    
    return emd

def get_emd_distance(distribution1, distribution2):
    # Calculate the label distributions for the two dataloaders

    # Convert the distributions to arrays (in the order of labels 0 to 9)
    distribution1_array = np.array([distribution1.get(i, 0) for i in range(10)])
    distribution2_array = np.array([distribution2.get(i, 0) for i in range(10)])
    
    # Calculate Earth Mover's Distance (EMD) between the two distributions
    emd = wasserstein_distance(distribution1_array, distribution2_array)
    
    return emd
