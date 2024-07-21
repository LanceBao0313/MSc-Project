import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import load_checkpoint
from dataset import CIFAR10
from configuration import DATA_PATH, DEVICE, BATCH_SIZE

# Function to evaluate a model on the test dataset
def evaluate_model(model, test_loader, device):
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    f1_score = 2 * (accuracy * 1) / (accuracy + 1)
    return accuracy, f1_score

# Main evaluation function
def evaluate_all_models(models_folder, test_loader, device='cuda'):
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pth') and 'device' in f]
    total_accuracy = 0.0
    total_f1 = 0.0
    num_models = len(model_files)

    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        model = load_checkpoint(model_path)
        accuracy, f1 = evaluate_model(model, test_loader, device)
        print(f'Accuracy of model {model_file}: {accuracy:.4f}, F1 score: {f1:.4f}')
        total_accuracy += accuracy
        total_f1 += f1

    average_accuracy = total_accuracy / num_models
    average_f1 = total_f1 / num_models
    print(f'Average accuracy of all models: {average_accuracy:.4f}')
    print(f'Average F1 score of all models: {average_f1:.4f}')
    return average_accuracy, average_f1

def run_evaluation():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    test_dataset = CIFAR10(root= DATA_PATH, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Folder containing the model weights (.pth files)
    models_folder = './checkpoints'

    # Evaluate all models and get the average accuracy
    average_accuracy, average_f1 = evaluate_all_models(models_folder, test_loader, device=DEVICE)

if __name__ == '__main__':
    
    run_evaluation()