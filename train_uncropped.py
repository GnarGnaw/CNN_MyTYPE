import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model import CelebA_CNN
import glob
import re


class CelebADataset(Dataset):
    def __init__(self, attr_path, land_path, partition_path, img_dir, split='train', transform=None):
        with open(attr_path, 'r') as f:
            lines = f.readlines()
            attr_names = ['image_id'] + lines[1].strip().split()

        with open(land_path, 'r') as f:
            lines = f.readlines()
            land_names = ['image_id'] + lines[1].strip().split()

        # Load official partitions [cite: 10, 24]
        partition_df = pd.read_csv(partition_path, sep=r'\s+', skiprows=0, names=['image_id', 'status'],
                                   engine='python')

        attr_df = pd.read_csv(attr_path, sep=r'\s+', skiprows=2, names=attr_names, engine='python')
        land_df = pd.read_csv(land_path, sep=r'\s+', skiprows=2, names=land_names, engine='python')

        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_status = split_map[split]
        allowed_ids = partition_df[partition_df['status'] == target_status]['image_id'].values

        self.attr_df = attr_df[attr_df['image_id'].isin(allowed_ids)].reset_index(drop=True)
        self.land_df = land_df[land_df['image_id'].isin(allowed_ids)].reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.filenames = self.attr_df['image_id'].values
        print(f"Dataset successfully loaded: {split.upper()} set with {len(self.filenames)} images.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        labels = self.attr_df.iloc[idx, 1:].values.astype(float)
        labels = (labels + 1) / 2

        # Landmark locations based on in-the-wild coordinates [cite: 18]
        landmarks = self.land_df.iloc[idx, 1:].values.astype(float)

        # Normalize to percentages of full photo for "In-The-Wild" training
        landmarks[0::2] /= orig_w
        landmarks[1::2] /= orig_h

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(landmarks, dtype=torch.float32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CelebA_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 1. Resumption Logic
    start_epoch = 0
    checkpoint_files = glob.glob("uncropped_model_epoch_*.pth")
    if checkpoint_files:
        epochs_found = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in checkpoint_files]
        last_epoch = max(epochs_found)
        checkpoint_path = f"uncropped_model_epoch_{last_epoch}.pth"
        print(f"Found checkpoint: {checkpoint_path}. Resuming from Epoch {last_epoch + 1} on {device}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_epoch = last_epoch
    else:
        print("No checkpoints found. Starting training from scratch.")

    # 2. Setup Data
    transform = transforms.Compose([
        transforms.Resize((218, 178)),  # Resize according to align&cropped standards [cite: 12]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load official partitions [cite: 10, 24]
    train_dataset = CelebADataset('attr.txt', 'landmarks.txt', 'list_eval_partition.txt', 'females_uncropped',
                                  split='train', transform=transform)
    val_dataset = CelebADataset('attr.txt', 'landmarks.txt', 'list_eval_partition.txt', 'females_uncropped',
                                split='val', transform=transform)

    # Create BOTH loaders here
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    criterion_attr = nn.BCEWithLogitsLoss()
    criterion_land = nn.MSELoss()

    # 3. Training Loop
    total_epochs = 20
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0.0
        print(f"\n--- Starting Epoch {epoch + 1} on {device}---")

        for i, (images, labels, landmarks) in enumerate(train_loader):
            images, labels, landmarks = images.to(device), labels.to(device), landmarks.to(device)

            optimizer.zero_grad()
            attr_preds, land_preds = model(images)

            loss_attr = criterion_attr(attr_preds, labels)
            loss_land = criterion_land(land_preds, landmarks)
            total_loss = loss_attr + (loss_land * 1000)

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {i}/{len(train_loader)} | Loss: {total_loss.item():.4f}")

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        print(f"Running Validation for Epoch {epoch + 1}...")
        with torch.no_grad():
            for images, labels, landmarks in val_loader:
                images, labels, landmarks = images.to(device), labels.to(device), landmarks.to(device)
                attr_preds, land_preds = model(images)

                v_loss_attr = criterion_attr(attr_preds, labels)
                v_loss_land = criterion_land(land_preds, landmarks)
                v_total_loss = v_loss_attr + (v_loss_land * 1000)
                val_loss += v_total_loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"\nEpoch {epoch + 1} Summary | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save Checkpoint
        save_name = f"uncropped_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved: {save_name}")