import torch
from model import CelebA_CNN

checkpoint = torch.load('celeba_model_epoch_4.pth')

# Check if it's a state_dict or a full model
if isinstance(checkpoint, dict):
    print("Checkpoint is a valid State Dict.")
    # Look at a few weights to see if they are non-zero/valid
    first_layer_key = next(iter(checkpoint))
    print(f"Sample weights from {first_layer_key}: {checkpoint[first_layer_key][0][0]}")
else:
    print("File is a full model object.")

print("Epoch 5 appears healthy.")