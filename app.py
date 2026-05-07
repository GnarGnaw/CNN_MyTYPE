import streamlit as st
from PIL import Image, ImageDraw
import torch
import os
import glob
import random
from model import CelebA_CNN
from torchvision import transforms

st.set_page_config(page_title="Find My Type AI", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "all_images" not in st.session_state:
    FEMALE_DIR = "females"
    images = sorted(glob.glob(os.path.join(FEMALE_DIR, "*.jpg")))
    random.shuffle(images)
    st.session_state.all_images = images

if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0
if "liked_vectors" not in st.session_state:
    st.session_state.liked_vectors = []
if "disliked_vectors" not in st.session_state:
    st.session_state.disliked_vectors = []

@st.cache_resource
def load_model():
    model = CelebA_CNN().to(device)
    checkpoint = torch.load("full_model_epoch_6.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()

@st.cache_resource
def load_database():
    if os.path.exists("face_database.pt"):
        return torch.load("face_database.pt")
    return {}

db = load_database()

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

transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tab1, tab2 = st.tabs(["🔥 Swiper", "🔍 Analyze Photo"])

with tab1:
    if not st.session_state.all_images:
        st.error("No images found.")
    else:
        current_img_path = st.session_state.all_images[st.session_state.img_idx]
        img_name = os.path.basename(current_img_path)

        col1, col2 = st.columns([1, 2])

        with col1:
            display_img = Image.open(current_img_path).convert('RGB')
            st.image(display_img, width='stretch')

        with col2:
            st.write(f"### Profile: {img_name}")

            input_tensor = transform(display_img).unsqueeze(0).to(device)
            with torch.no_grad():
                attr_preds, land_preds = model(input_tensor)

            probs = torch.sigmoid(attr_preds).squeeze().cpu().numpy()
            lands = land_preds.squeeze().cpu().numpy()
            face_vector = torch.cat([torch.tensor(probs), torch.tensor(lands)])

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("❌ Swipe Left", width='stretch'):
                    st.session_state.disliked_vectors.append(face_vector)
                    st.session_state.img_idx = (st.session_state.img_idx + 1) % len(st.session_state.all_images)
                    st.rerun()
            with b_col2:
                if st.button("✅ Swipe Right", width='stretch'):
                    st.session_state.liked_vectors.append(face_vector)
                    st.session_state.img_idx = (st.session_state.img_idx + 1) % len(st.session_state.all_images)
                    st.rerun()

            with st.expander("🛠️ Debug: AI Attribute Predictions", expanded=True):
                display_attrs = [(n, p) for n, p in zip(ATTR_NAMES, probs) if n != "No_Beard"]
                top_debug = sorted(display_attrs, key=lambda x: x[1], reverse=True)[:6]
                for name, p in top_debug:
                    st.text(f"{name:<20} {p * 100:>5.1f}%")

            st.divider()

            if st.button("🌟 FIND MY MATCH", type="primary", width='stretch'):
                if len(st.session_state.liked_vectors) < 1:
                    st.warning("Swipe right on at least one person first!")
                elif not db:
                    st.error("Database not found! Please run build_db.py first.")
                else:
                    status_text = st.empty()
                    progress_bar = st.progress(0)

                    status_text.text("🧠 Calculating your target 'Type' profile...")

                    liked_avg = torch.mean(torch.stack(st.session_state.liked_vectors), dim=0)

                    if st.session_state.disliked_vectors:
                        disliked_avg = torch.mean(torch.stack(st.session_state.disliked_vectors), dim=0)
                        target_vector = liked_avg - (disliked_avg * 0.5)
                        target_vector = torch.clamp(target_vector, min=0.0, max=1.0)
                    else:
                        target_vector = liked_avg

                    ignore_indices = [0, 16, 20, 22, 24, 30]
                    keep_indices = [i for i in range(40) if i not in ignore_indices]

                    attr_target_filtered = target_vector[keep_indices]
                    attr_target_centered = attr_target_filtered - 0.5
                    target_lands = target_vector[-10:]

                    status_text.text("⚡ Scanning database for your absolute perfect match...")

                    best_match_path = None
                    highest_score = -1.0
                    cos = torch.nn.CosineSimilarity(dim=0)

                    total_items = len(db)

                    for idx, (path, vector) in enumerate(db.items()):
                        if idx % (max(1, total_items // 100)) == 0:
                            progress_bar.progress(min(100, int((idx / total_items) * 100)))

                        is_male_prob = vector[20].item()

                        attr_current_filtered = vector[keep_indices]
                        attr_current_centered = attr_current_filtered - 0.5

                        struct_sim_raw = cos(target_lands, vector[-10:]).item()
                        struct_sim = max(0.0, (struct_sim_raw - 0.95) / 0.05)

                        attr_sim_raw = cos(attr_target_centered, attr_current_centered).item()
                        attr_sim = max(0.0, attr_sim_raw)

                        combined_sim = (attr_sim * 0.9) + (struct_sim * 0.1)
                        final_score = combined_sim * 100

                        if is_male_prob > 0.4:
                            final_score = final_score * (1 - is_male_prob)

                        if final_score > highest_score:
                            highest_score = final_score
                            best_match_path = path

                    progress_bar.empty()
                    status_text.success(f"✅ Match Found!")

                    st.divider()
                    st.subheader(f"🎉 Your Best Match Found! ({highest_score:.1f}% Compatibility)")

                    clean_filename = best_match_path.replace("\\", "/").split("/")[-1]
                    safe_path = os.path.join("females", clean_filename)

                    match_img = Image.open(safe_path).convert('RGB')
                    st.image(match_img, caption=f"Top Match: {clean_filename}", width='stretch')
                    st.balloons()

with tab2:
    st.subheader("Manual Analysis & Compatibility Check")
    uploaded_file = st.file_uploader("Upload a photo to check compatibility...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        orig_w, orig_h = img.size

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            attr_preds, land_preds = model(input_tensor)

        probs = torch.sigmoid(attr_preds).squeeze().cpu().numpy()
        lands = land_preds.squeeze().cpu().numpy()
        current_vector = torch.cat([torch.tensor(probs), torch.tensor(lands)])

        draw = ImageDraw.Draw(img)
        line_w = max(2, int(orig_w * 0.002))
        pts = [(lands[i * 2] * orig_w, lands[i * 2 + 1] * orig_h) for i in range(5)]

        e_cx, e_cy = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
        e_dist = abs(pts[1][0] - pts[0][0])
        draw.rectangle([e_cx - e_dist, e_cy - e_dist * 0.2, e_cx + e_dist, e_cy + e_dist * 0.2], outline="lime",
                       width=line_w)

        draw.rectangle([pts[2][0] - orig_w * 0.04, pts[2][1] - orig_h * 0.025, pts[2][0] + orig_w * 0.04,
                        pts[2][1] + orig_h * 0.025], outline="yellow", width=line_w)

        m_cx, m_cy = (pts[3][0] + pts[4][0]) / 2, (pts[3][1] + pts[4][1]) / 2
        m_dist = abs(pts[4][0] - pts[3][0])
        draw.rectangle([m_cx - m_dist * 0.7, m_cy - m_dist * 0.2, m_cx + m_dist * 0.7, m_cy + m_dist * 0.2],
                       outline="red", width=line_w)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(img, caption="AI Vision Analysis", width='stretch')

        with col2:
            st.write("### 💘 Compatibility Check")

            if not st.session_state.liked_vectors:
                st.warning("Please swipe right on some people in the 'Swiper' tab first!")
            else:
                liked_avg = torch.mean(torch.stack(st.session_state.liked_vectors), dim=0)

                if st.session_state.disliked_vectors:
                    disliked_avg = torch.mean(torch.stack(st.session_state.disliked_vectors), dim=0)
                    target_vector = liked_avg - (disliked_avg * 0.5)
                    target_vector = torch.clamp(target_vector, min=0.0, max=1.0)
                else:
                    target_vector = liked_avg

                male_idx = 20
                is_male_prob = probs[male_idx]

                ignore_indices = [0, 16, 20, 22, 24, 30]
                keep_indices = [i for i in range(40) if i not in ignore_indices]

                attr_target_filtered = target_vector[keep_indices]
                attr_current_filtered = torch.tensor(probs)[keep_indices]

                attr_target_centered = attr_target_filtered - 0.5
                attr_current_centered = attr_current_filtered - 0.5

                cos = torch.nn.CosineSimilarity(dim=0)

                struct_sim_raw = cos(target_vector[-10:], torch.tensor(lands)).item()
                struct_sim = max(0.0, (struct_sim_raw - 0.95) / 0.05)

                attr_sim_raw = cos(attr_target_centered, attr_current_centered).item()
                attr_sim = max(0.0, attr_sim_raw)

                combined_sim = (attr_sim * 0.99) + (struct_sim * 0.01)
                final_score = combined_sim * 100

                if is_male_prob > 0.4:
                    final_score = final_score * (1 - is_male_prob)

                if is_male_prob > 0.6:
                    st.warning(
                        f"Low Compatibility: High probability of masculine features ({is_male_prob * 100:.1f}%).")
                elif final_score > 90:
                    st.success(f"**MATCH ALERT!** Compatibility: **{final_score:.1f}%**")
                    st.balloons()
                else:
                    st.info(f"Compatibility Score: **{final_score:.1f}%**")

            with st.expander("🛠️ Detailed Attributes", expanded=True):
                st.markdown(f"**Male Probability:** `{probs[20] * 100:.1f}%`")

                attr_data = [(n, p) for n, p in zip(ATTR_NAMES, probs) if n != "No_Beard"]
                top_attrs = sorted(attr_data, key=lambda x: x[1], reverse=True)[:10]
                for name, p in top_attrs:
                    st.text(f"{name.replace('_', ' '):<20} {p * 100:>5.1f}%")