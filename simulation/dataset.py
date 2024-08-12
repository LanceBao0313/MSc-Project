import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from collections import defaultdict
from configuration import BATCH_SIZE, NUMBER_OF_DEVICES, NUM_OF_CLASSES, NON_IID_ALPHA, RANDOM_SEED
from typing import List, Tuple

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

def get_CIFAR10_dataloader(root, train=True):
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    cifar10 = CIFAR10(root=root, train=train, download=False, transform=transform)
    
    # Create DataLoader
    dataloader = DataLoader(cifar10, batch_size=BATCH_SIZE, shuffle=True)
    
    return dataloader


def get_nonIID_dataloader(root, train=True):
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    batch_size: int
        Batch size for the data loaders
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[DataLoader], DataLoader]
        The list of data loaders for each client, the test data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    cifar10 = CIFAR10(root=root, train=train, download=False, transform=transform)

    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(RANDOM_SEED)

    # get the targets
    tmp_t = cifar10.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List[int]] = [[] for _ in range(NUMBER_OF_DEVICES)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(NON_IID_ALPHA, NUMBER_OF_DEVICES))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / NUMBER_OF_DEVICES)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(cifar10, idxs) for idxs in idx_clients]
   
    # Create data loaders for each client
    client_dataloaders = [DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) for dataset in trainsets_per_client]

    return client_dataloaders