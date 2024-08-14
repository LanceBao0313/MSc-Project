import torch.nn as nn
import torch
from configuration import NUM_EPOCHS, NUM_OF_CLASSES, DEVICE, RANDOM_SEED
from mcunet.mcunet.model_zoo import build_model
import os
import numpy as np
import math
from collections import OrderedDict
from typing import List

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

def train(model, train_loader, optimizer, checkpoint_path, num_samples, emd):
	# model, optimizer = load_checkpoint(checkpoint_path)
	criterion = nn.CrossEntropyLoss()
	model.to(DEVICE)
	model.train()

	for state in optimizer.state.values():
		for k, v in state.items():
			if isinstance(v, torch.Tensor):
				state[k] = v.to(DEVICE)

	# Training loop
	correct, total, epoch_loss = 0, 0, 0.0
	for epoch in range(NUM_EPOCHS):
		correct, total, epoch_loss = 0, 0, 0.0
		
		for inputs, labels in train_loader:
			# if counter >= 30:
			# 	break
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			# print(f"labels: {labels[:10]}")
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss
			total += labels.size(0)
			correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
		epoch_loss /= len(train_loader.dataset)
		epoch_acc = correct / total
		# print(f"Epoch {epoch+1}:loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}, num_samples: {num_samples}")
	save_checkpoint(model, optimizer, num_samples, emd, checkpoint_path)
	# print(model.classifier[0].weight.data)
	return epoch_loss, epoch_acc

def replace_fc_layer(model, num_classes):
	num_ftrs = model.classifier.in_features
	# Freeze all layers
	for param in model.parameters():
		param.requires_grad = False
	classifier  = nn.Sequential(
		nn.Linear(num_ftrs, 128),
		nn.ReLU(),
		#nn.Dropout(0.2),  
		nn.Linear(128, 64),
		nn.ReLU(),
		#nn.Dropout(0.2),  
		nn.Linear(64, num_classes)
	)
	model.classifier = classifier #nn.Linear(num_ftrs, num_classes)
	model.classifier.requires_grad = True
	return model

def save_checkpoint(model, optimizer, num_samples, emd, filepath):
	state = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'num_samples': num_samples,
		'emd': emd
	}
	torch.save(state, filepath)

# Function to load model and optimizer state
def load_checkpoint(filepath):
	model, resolution, description = build_model(net_id="mcunet-in3", pretrained=True)
	model = replace_fc_layer(model, NUM_OF_CLASSES)
	optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay=1e-3)
	if os.path.isfile(filepath):
		checkpoint = torch.load(filepath)
		model.load_state_dict(checkpoint['model_state_dict'])
		# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])	
		if 'num_samples' in checkpoint:
			num_samples = checkpoint['num_samples']
			return model, num_samples, optimizer
		else:
			return model, 0, optimizer

# def fedAvg(weight_list, sample_list):
#     """Compute the weighted average of weights according to the number of samples.

#     Parameters
#     ----------
#     weight_list : List[List[torch.Tensor]]
#         List of weights from different clients
#     sample_list : List[int]
#         List of the number of samples corresponding to each client's weights

#     Returns
#     -------
#     List[torch.Tensor]
#         The weighted average of the weights
#     """
#     # Ensure inputs are valid
#     assert len(weight_list) == len(sample_list), "The length of weight_list and sample_list must be equal"
    
#     # Total number of samples
#     total_samples = sum(sample_list)
    
#     # Initialize the average weights with zeros
#     avg_weights = [torch.zeros_like(w) for w in weight_list[0]]
    
#     # Calculate the weighted average
#     for weights, samples in zip(weight_list, sample_list):
#         for i, w in enumerate(weights):
#             avg_weights[i] += w * (samples / total_samples)
    
#     return avg_weights

def running_fedAvg(weight_1, weight_2, counter_1, counter_2, num_samples_1, num_samples_2):
    # combined_average = (running_avg_A * count_A + running_avg_B * count_B) / (count_A + count_B)
	gcd_counter = math.gcd(counter_1, counter_2)
	gcd_samples = math.gcd(num_samples_1, num_samples_2)
	# print(f"counter_1: {counter_1}, counter_2: {counter_2}")
	counter_1 = counter_1 // gcd_counter
	counter_2 = counter_2 // gcd_counter
	num_samples_1 = num_samples_1 // gcd_samples
	num_samples_2 = num_samples_2 // gcd_samples

	avg_weights = []
	for w1, w2 in zip(weight_1, weight_2):
		if not isinstance(w1, torch.Tensor):
			w1 = torch.tensor(w1, dtype=torch.float32)
		if not isinstance(w2, torch.Tensor):
			w2 = torch.tensor(w2, dtype=torch.float32)
			
		avg_weight = (w1*counter_1*num_samples_1+w2*counter_2*num_samples_2)/(counter_1*num_samples_1+counter_2*num_samples_2)
		avg_weights.append(avg_weight)
	return avg_weights

# def cumulative_fedAvg(total_weight, weight, total_samples, num_samples):
#     # combined_average = (running_avg_A * count_A + running_avg_B * count_B) / (count_A + count_B)

# 	result_weights = []
# 	for w1, w2 in zip(total_weight, weight):
			
# 		combined_weight = w1 + w2*num_samples
# 		result_weights.append(combined_weight)
# 	total_samples += num_samples
# 	return result_weights, total_samples

def cumulative_fedAvg(total_weight, weight, total_samples, num_samples):
    result_weights = []
    for w1, w2 in zip(total_weight, weight):
        combined_weight = w1 + w2 * num_samples
        result_weights.append(combined_weight)
    total_samples += num_samples
    return result_weights, total_samples


def reset_classifier_weights(model):
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def extract_classifier_weights(model):
    classifier = model.classifier
    weights = []
    for layer in classifier:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data.clone())
            weights.append(layer.bias.data.clone())
    return weights

def set_weights(model, weights):
    classifier = model.classifier
    weight_idx = 0
    for layer in classifier:
        if isinstance(layer, nn.Linear):
            layer.weight.data = weights[weight_idx].clone()
            layer.bias.data = weights[weight_idx + 1].clone()
            weight_idx += 2

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.classifier.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.classifier.load_state_dict(state_dict, strict=True)


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.classifier.state_dict().items()]