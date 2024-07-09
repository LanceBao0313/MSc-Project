import torch.nn as nn
import torch
from configuration import NUM_EPOCHS

def train(model, train_loader, criterion, optimizer, device):
	# Training loop
	for epoch in range(NUM_EPOCHS):
		model.train()
		running_loss = 0.0
		counter = 0
		
		for inputs, labels in train_loader:
			if counter >= 30:
				break
			inputs, labels = inputs.to(device), labels.to(device)
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			counter += 1
		
		epoch_loss = running_loss / len(train_loader)
		print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

	print("Training completed!")

	# Save the model
	# torch.save(model.state_dict(), 'fine_tuned_model.pth')

def replace_fc_layer(model, num_classes):
	num_ftrs = model.classifier.in_features
	# Freeze all layers
	for param in model.parameters():
		param.requires_grad = False
	model.classifier = nn.Linear(num_ftrs, num_classes)
	model.classifier.requires_grad = True
	return model


