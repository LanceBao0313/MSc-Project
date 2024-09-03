import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataset import get_nonIID_dataloader, get_IID_dataloader
from model import reset_classifier_weights
from torchvision.datasets import CIFAR10
import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from configuration import DATA_PATH, DEVICE, BATCH_SIZE, NUMBER_OF_DEVICES, NUM_EPOCHS, NUM_OF_CLASSES, RANDOM_SEED
import ray
from collections import OrderedDict
from typing import List, Tuple
import os
from mcunet.mcunet.model_zoo import build_model
from model import replace_fc_layer, reset_classifier_weights
import random
import time
import csv

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

start_time = time.time()

def save_checkpoint(model, optimizer, filepath):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filepath)

# Function to load model and optimizer state
def load_checkpoint(filepath):
    model, resolution, description = build_model(net_id="mcunet-in3", pretrained=True)
    model = replace_fc_layer(model, NUM_OF_CLASSES)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # print(f"Loaded {filepath} model")
        # print(model.classifier[0].weight.data)
    else:
        checkpoint = torch.load("./checkpoints/final_baseline.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        reset_classifier_weights(model)
        # print(model.classifier[0].weight.data)
        # print("Loaded baseline model")
        # optimizer = None
    return model, optimizer

# Function to load datasets
def load_datasets(partition_id: int):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # client_dataloaders = get_nonIID_dataloader(DATA_PATH, train=True)
    client_dataloaders = get_IID_dataloader(DATA_PATH)
    trainloader = client_dataloaders[partition_id]

    # Load CIFAR-10 dataset
    test_dataset = CIFAR10(root= DATA_PATH, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, test_loader

def train(id, model, optimizer, trainloader, testloader, epochs=1, verbose=False):
    # return
    # optimizer = optim.Adam(model.classifier.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(DEVICE)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # for batch in trainloader:
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # print("labels", labels[:10])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        if verbose:
            print(f"ID: {id} Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}, num_samples: {len(trainloader.dataset)}")
    # Evaluate on the test set
    # test_accuracy, test_f1 = test(model, testloader)

    # # Calculate wall clock time
    end_time = time.time()
    wall_clock_time = end_time - start_time
    print(f"{id}, {epoch_loss}, {0}, {0}, {len(trainloader.dataset)}, {wall_clock_time}")
    # return model, optimizer

def test(model, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # loss /= len(testloader.dataset)
    accuracy = correct / total
    f1_score = 2 * (accuracy * 1) / (accuracy + 1)
    return accuracy, f1_score

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.classifier.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.classifier.load_state_dict(state_dict, strict=True)


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.classifier.state_dict().items()]

class CifarClient(fl.client.NumPyClient):
    def __init__(self, id, train_loader, test_loader):
        self.id = id
        self.model, self.optimizer = load_checkpoint(f"CFL_{self.id}.pth")
        # self.model, self.optimizer = load_checkpoint_2(f"./checkpoints/device_{self.id}_checkpoint.pth")

        self.model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(self.id, self.model, self.optimizer, self.train_loader, self.test_loader, epochs=NUM_EPOCHS)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        # print(f"ID: {self.id} Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    partition_id = context.node_config["partition-id"]
    trainloader, testloader = load_datasets(partition_id=partition_id)
    # print(f"Client {partition_id} loaded {len(trainloader.dataset)} samples, {len(testloader.dataset)} test samples")
    return CifarClient(partition_id, trainloader, testloader).to_client()

def server_fn(context: Context) -> ServerAppComponents:

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=6)

    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 10 clients for training
        min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
        min_available_clients=2,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(strategy=strategy, config=config)

# Start the federated learning process
if __name__ == "__main__":
    # Split the training dataset into multiple clients
    num_clients = 100

    client = ClientApp(client_fn=client_fn)
    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)
    # Initialize clients

    # When running on GPU, assign an entire GPU for each client
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    ray.init(runtime_env={"working_dir": ".", "excludes": ["data"]})

    start_time = time.time()

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
