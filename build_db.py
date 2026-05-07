import torch
import os
import glob
from PIL import Image
from torchvision import transforms
from model import CelebA_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your BEST model (We determined Epoch 2 was the sweet spot before overfitting)
model = CelebA_CNN().to(device)
model.load_state_dict(torch.load("graphing_model_epoch_2.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_paths = glob.glob("females/*.jpg")
database = {}

print(f"Building Match Database for {len(image_paths)} images...")

with torch.no_grad():
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        attr_preds, land_preds = model(tensor)
        probs = torch.sigmoid(attr_preds).squeeze().cpu()
        lands = land_preds.squeeze().cpu()

        # Combine into one 50-number tensor
        vector = torch.cat([probs, lands])
        database[path] = vector

        if i % 500 == 0:
            print(f"Processed {i}/{len(image_paths)} images...")

torch.save(database, "face_database.pt")
print("✅ Success! Database saved as face_database.pt")