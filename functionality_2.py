import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

st.title("👁️ AI Face Analyzer")
st.write("Upload a photo and the CNN will predict attributes and landmarks.")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()

    # 1. Pre-processing (Match the CNN training requirements)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    # 2. Inference (Loading a dummy model for structure)
    # In a real project, you'd do: model.load_state_dict(torch.load('model.pth'))
    model = CelebA_CNN()
    model.eval()

    with torch.no_grad():
        attr_logits, land_preds = model(input_tensor)

    # 3. Post-processing Attributes
    probs = torch.sigmoid(attr_logits).squeeze().numpy()
    # We'll use our attr_names from the engine
    attr_names = st.session_state.rec.attr_names

    # 4. Post-processing Landmarks (Rescale back to image size)
    # This assumes model was trained on normalized [0,1] coordinates
    coords = land_preds.squeeze().numpy()
    w, h = display_img.size

    draw = ImageDraw.Draw(display_img)
    for i in range(0, 10, 2):
        lx, ly = coords[i] * w, coords[i + 1] * h
        draw.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], fill='cyan', outline='white')

    # UI Layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="AI Detection", use_container_width=True)

    with col2:
        st.subheader("Predicted Traits")
        # Show traits with > 50% probability
        found_traits = [attr_names[i] for i, p in enumerate(probs) if p > 0.5]
        if found_traits:
            for t in found_traits:
                st.write(f"✅ {t}")
        else:
            st.write("No strong traits detected.")