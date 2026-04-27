import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import glob
import re
from model import CelebA_CNN

# 1. Setup Device and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CelebA_CNN()
checkpoint_files = glob.glob("celeba_model_epoch_*.pth")
if not checkpoint_files:
    print("Error: No .pth checkpoints found in the current directory.")
    exit()

# Load the latest epoch automatically
latest_epoch = max([int(re.search(r'epoch_(\d+)', f).group(1)) for f in checkpoint_files])
checkpoint_path = f"celeba_model_epoch_{latest_epoch}.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()
print(f"Model loaded from {checkpoint_path}")

# 2. Configuration
TARGET_W, TARGET_H = 178, 218
input_folder = 'tests'
output_folder = 'test_results'
os.makedirs(output_folder, exist_ok=True)

# Training transforms (must match train.py exactly)
transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def process_image(image_path, filename):
    # Load original image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    # Prepare input for model
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        attr_preds, land_preds = model(input_tensor)

    # Convert landmarks to numpy
    # Prediction is in the 0.0 - 1.0 range (because of / 218.0 in training)
    raw = land_preds.squeeze().cpu().numpy()

    # Mapping logic
    # Scale_x/y mirrors your draw_rectangle_pillow logic
    scale_x = orig_w / TARGET_W
    scale_y = orig_h / TARGET_H

    labels = ["L_Eye", "R_Eye", "Nose", "L_Mouth", "R_Mouth"]
    colors = ["lime", "lime", "yellow", "red", "red"]

    draw = ImageDraw.Draw(img)
    # Dynamic circle radius (approx 1.5% of width)
    r = max(8, int(orig_w * 0.015))

    print(f"\n--- DEBUG: {filename} ---")

    for i in range(5):
        # The raw values from the model
        raw_x = raw[i * 2]
        raw_y = raw[i * 2 + 1]

        # Step 1: Denormalize back to the 178x218 pixel space
        # We multiply by 218 because 'landmarks / 218.0' was used in train.py
        tx = raw_x * 178.0
        ty = raw_y * 218.0

        # Step 2: Scale to the high-res original image
        # Using your scale_x/y logic
        final_x = tx * scale_x
        final_y = ty * scale_y

        print(f"{labels[i]:<7} | Raw: ({raw_x: .4f}, {raw_y: .4f}) | Grid: ({tx: .1f}, {ty: .1f})")

        # Step 3: Draw the circles
        # Left-top and bottom-right corners for the ellipse
        box = [final_x - r, final_y - r, final_x + r, final_y + r]
        draw.ellipse(box, outline=colors[i], width=6)

    # Save output
    save_path = os.path.join(output_folder, f"RESULT_{filename}")
    img.save(save_path)
    print(f"Saved result to: {save_path}")


# 3. Main Loop
if __name__ == "__main__":
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in '{input_folder}' folder.")
    else:
        for filename in files:
            process_image(os.path.join(input_folder, filename), filename)
        print("\nAll images processed successfully.")