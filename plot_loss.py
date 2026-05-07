import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Data structures
epoch_batch_losses = defaultdict(list)
epoch_numbers = []
epoch_train_losses = []
epoch_val_losses = []

# Read and parse the text file
with open("data_train.txt", "r") as file:
    for line in file:
        # Match Batch-level Training Loss
        # Example: Epoch 1 | Batch 100/2954 | Loss: 7.5368
        batch_match = re.search(r'Epoch (\d+) \| Batch \d+/\d+ \| Loss: ([\d.]+)', line)
        if batch_match:
            epoch_num = int(batch_match.group(1))
            loss_val = float(batch_match.group(2))
            epoch_batch_losses[epoch_num].append(loss_val)

        # Match Epoch Summary Loss
        # Example: Epoch 1 Summary | Train Loss: 4.1234 | Val Loss: 4.5678
        summary_match = re.search(r'Epoch (\d+) Summary \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
        if summary_match:
            epoch_numbers.append(int(summary_match.group(1)))
            epoch_train_losses.append(float(summary_match.group(2)))
            epoch_val_losses.append(float(summary_match.group(3)))

# Create the plots
plt.figure(figsize=(14, 5))

# Plot 1: Granular Batch Loss (Overlaid by Epoch)
plt.subplot(1, 2, 1)

# Plot each epoch's batch losses as a separate line
for epoch_num, losses in sorted(epoch_batch_losses.items()):
    plt.plot(losses, label=f"Epoch {epoch_num}", alpha=0.8, linewidth=1.5)

plt.title("Intra-Epoch Training Loss Comparison")
plt.xlabel("Training Steps (per 100 batches)")
plt.ylabel("Total Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot 2: Train vs Val Loss (The Overfitting Check)
if epoch_numbers:
    plt.subplot(1, 2, 2)
    plt.plot(epoch_numbers, epoch_train_losses, label="Train Loss", marker='o', linewidth=2)
    plt.plot(epoch_numbers, epoch_val_losses, label="Validation Loss", marker='o', linewidth=2)
    plt.title("Epoch Summary: Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    # Force the X-axis to only show whole numbers for epochs
    plt.xticks(epoch_numbers)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
else:
    print("No 'Epoch Summary' lines found. Could not generate the Train vs Val graph.")

plt.tight_layout()
plt.show()