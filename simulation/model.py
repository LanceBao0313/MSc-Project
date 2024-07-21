import torch.nn as nn
import torch
from configuration import NUM_EPOCHS, NUM_OF_CLASSES, DEVICE
from mcunet.mcunet.model_zoo import build_model
import os

def train(model, train_loader, criterion, optimizer, checkpoint_path):
	# model, optimizer = load_checkpoint(checkpoint_path)
	model.to(DEVICE)
	# model.train()
	# Training loop
	for epoch in range(NUM_EPOCHS):
		running_loss = 0.0
		counter = 0
		
		for inputs, labels in train_loader:
			# if counter >= 30:
			# 	break
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			counter += 1
		
		epoch_loss = running_loss / len(train_loader)
		#print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
	save_checkpoint(model, optimizer, checkpoint_path)
	#print("Training completed!")
	# print(f"Loss: {epoch_loss:.4f}")


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
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	if os.path.isfile(filepath):
		checkpoint = torch.load(filepath)
		model.load_state_dict(checkpoint['model_state_dict'])
		# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])	
	return model#, optimizer

def fedAvg(weight_list, sample_list):
    """Compute the weighted average of weights according to the number of samples.

    Parameters
    ----------
    weight_list : List[List[torch.Tensor]]
        List of weights from different clients
    sample_list : List[int]
        List of the number of samples corresponding to each client's weights

    Returns
    -------
    List[torch.Tensor]
        The weighted average of the weights
    """
    # Ensure inputs are valid
    assert len(weight_list) == len(sample_list), "The length of weight_list and sample_list must be equal"
    
    # Total number of samples
    total_samples = sum(sample_list)
    
    # Initialize the average weights with zeros
    avg_weights = [torch.zeros_like(w) for w in weight_list[0]]
    
    # Calculate the weighted average
    for weights, samples in zip(weight_list, sample_list):
        for i, w in enumerate(weights):
            avg_weights[i] += w * (samples / total_samples)
    
    return avg_weights

# def fedAvg(weights1, weights2):
#     avg_weights = []
#     for w1, w2 in zip(weights1, weights2):
#         avg_weight = (w1 + w2) / 2
#         avg_weights.append(avg_weight)
#     return avg_weights


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