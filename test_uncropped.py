import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import re
from model import CelebA_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model
model = CelebA_CNN()
checkpoint_files = glob.glob("uncropped_model_epoch_*.pth")
if not checkpoint_files:
    print("No checkpoints found!")
    exit()

latest_epoch = max([int(re.search(r'epoch_(\d+)', f).group(1)) for f in checkpoint_files])
model.load_state_dict(torch.load(f"uncropped_model_epoch_{latest_epoch}.pth", map_location=device))
model.to(device).eval()

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

TARGET_W, TARGET_H = 178, 218

transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def process_combined_boxes(image_path, filename):
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        attr_preds, land_preds = model(input_tensor)

    probs = torch.sigmoid(attr_preds).squeeze().cpu().numpy()
    raw = land_preds.squeeze().cpu().numpy()

    # Landmark mapping
    pts = []
    for i in range(5):
        pts.append((raw[i * 2] * orig_w, raw[i * 2 + 1] * orig_h))

    l_eye, r_eye, nose, l_mouth, r_mouth = pts[0], pts[1], pts[2], pts[3], pts[4]

    draw = ImageDraw.Draw(img)

    # Settings
    line_w = max(2, int(orig_w * 0.002))
    try:
        font = ImageFont.truetype("arial.ttf", int(orig_w * 0.015))
    except:
        font = ImageFont.load_default()

    print(f"\n{'=' * 20} {filename} {'=' * 20}")
    attr_results = sorted(zip(ATTR_NAMES, probs), key=lambda x: x[1], reverse=True)
    for name, prob in attr_results[:8]:
        print(f"  - {name:<20}: {prob * 100:>5.1f}%")

    # --- 1. Eyes Box ---
    e_cx, e_cy = (l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2
    e_dist = abs(r_eye[0] - l_eye[0])
    eb_w, eb_h = e_dist * 1.8, e_dist * 0.45

    draw.rectangle([e_cx - eb_w / 2, e_cy - eb_h / 2, e_cx + eb_w / 2, e_cy + eb_h / 2],
                   outline="lime", width=line_w)
    draw.text((e_cx - eb_w / 2, e_cy - eb_h / 2 - (orig_h * 0.02)), "EYES", fill="lime", font=font)

    # --- 2. Nose Box ---
    nb_w, nb_h = orig_w * 0.08, orig_h * 0.05
    draw.rectangle([nose[0] - nb_w / 2, nose[1] - nb_h / 2, nose[0] + nb_w / 2, nose[1] + nb_h / 2],
                   outline="yellow", width=line_w)
    draw.text((nose[0] - nb_w / 2, nose[1] - nb_h / 2 - (orig_h * 0.02)), "NOSE", fill="yellow", font=font)

    # --- 3. Mouth Box ---
    m_cx, m_cy = (l_mouth[0] + r_mouth[0]) / 2, (l_mouth[1] + r_mouth[1]) / 2
    m_dist = abs(r_mouth[0] - l_mouth[0])
    mb_w, mb_h = m_dist * 1.5, m_dist * 0.4

    draw.rectangle([m_cx - mb_w / 2, m_cy - mb_h / 2, m_cx + mb_w / 2, m_cy + mb_h / 2],
                   outline="red", width=line_w)
    draw.text((m_cx - mb_w / 2, m_cy - mb_h / 2 - (orig_h * 0.02)), "MOUTH", fill="red", font=font)

    os.makedirs('test_results_final', exist_ok=True)
    img.save(f'test_results_final/RESULT_{filename}')


# Execution
input_folder = 'tests'
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_combined_boxes(os.path.join(input_folder, filename), filename)