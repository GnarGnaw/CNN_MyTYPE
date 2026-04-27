import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model import CelebA_CNN
import glob
import re


class CelebADataset(Dataset):
    def __init__(self, attr_path, land_path, img_dir, transform=None):
        with open(attr_path, 'r') as f:
            lines = f.readlines()
            attr_names = ['image_id'] + lines[1].strip().split()

        with open(land_path, 'r') as f:
            lines = f.readlines()
            land_names = ['image_id'] + lines[1].strip().split()

        attr_df = pd.read_csv(attr_path, sep=r'\s+', skiprows=2, names=attr_names, engine='python')
        land_df = pd.read_csv(land_path, sep=r'\s+', skiprows=2, names=land_names, engine='python')

        attr_df['image_id'] = attr_df['image_id'].astype(str)
        mask = attr_df['image_id'].str.lower().str.endswith('.jpg')
        self.attr_df = attr_df[mask].reset_index(drop=True)
        self.land_df = land_df[land_df['image_id'].isin(self.attr_df['image_id'])].reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.filenames = self.attr_df['image_id'].values
        print(f"Dataset successfully loaded with {len(self.filenames)} images.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        labels = self.attr_df.iloc[idx, 1:].values.astype(float)
        labels = (labels + 1) / 2

        landmarks = self.land_df.iloc[idx, 1:].values.astype(float)
        # Normalizing landmarks (Divide by original image size)
        landmarks = landmarks / 218.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(landmarks, dtype=torch.float32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CelebA_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 1. Resumption Logic
    start_epoch = 1
    checkpoint_files = glob.glob("celeba_model_epoch_*.pth")

    if checkpoint_files:
        epochs_found = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in checkpoint_files]
        last_epoch = max(epochs_found)
        checkpoint_path = f"celeba_model_epoch_{last_epoch}.pth"
        print(f"Found checkpoint: {checkpoint_path}. Resuming from Epoch {last_epoch + 1} on {device}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_epoch = last_epoch  # Loop starts at last_epoch (which saves as last_epoch + 1)
    else:
        print("No checkpoints found. Starting training from scratch.")

    # 2. Setup Data
    transform = transforms.Compose([
        transforms.Resize((218, 178)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = CelebADataset('attr.txt', 'landmarks_cropped.txt', 'females', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    criterion_attr = nn.BCEWithLogitsLoss()
    criterion_land = nn.MSELoss()

    # 3. Training Loop
    total_epochs = 20
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"\n--- Starting Epoch {epoch + 1} on {device}---")

        for i, (images, labels, landmarks) in enumerate(train_loader):
            images, labels, landmarks = images.to(device), labels.to(device), landmarks.to(device)

            optimizer.zero_grad()
            attr_preds, land_preds = model(images)

            # Loss Calculation
            loss_attr = criterion_attr(attr_preds, labels)
            loss_land = criterion_land(land_preds, landmarks)

            # THE FIX: Increase landmark importance.
            # If boxes don't move, increase 50 to 100 or 500.
            total_loss = loss_attr + (loss_land * 1000)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {i}/{len(train_loader)} | Loss: {total_loss.item():.4f}")

        # Save Checkpoint
        save_name = f"celeba_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved: {save_name}")