import streamlit as st
from PIL import Image
import os
import pandas as pd
from engine import Recommender

st.set_page_config(page_title="Celeb-Match Pro", layout="wide")

# Initialize the global dictionary of profiles if it doesn't exist
if 'profiles' not in st.session_state:
    st.session_state.profiles = {}

# LANDING PAGE: Profile Selection
if 'current_user' not in st.session_state:
    st.title("📂 Welcome to Celeb-Match")
    st.write("Please enter a name to start or resume your personalized recommendation test.")

    user_input = st.text_input("User Profile Name:", placeholder="e.g., Alex, Tester_1")

    if st.button("Start Swiping"):
        if user_input.strip() != "":
            st.session_state.current_user = user_input.strip()

            # Create a new engine for this user if it's their first time
            if st.session_state.current_user not in st.session_state.profiles:
                with st.spinner("Initializing neural engine for new profile..."):
                    st.session_state.profiles[st.session_state.current_user] = {
                        'rec': Recommender('attr.txt', 'landmarks.txt'),
                        'current_img': None,
                        'current_idx': None,
                        'match_traits': None,
                        'match_score': 0
                    }
                # Pick the first image for the new user
                rec = st.session_state.profiles[st.session_state.current_user]['rec']
                fname, fidx = rec.get_next()
                st.session_state.profiles[st.session_state.current_user]['current_img'] = fname
                st.session_state.profiles[st.session_state.current_user]['current_idx'] = fidx

            st.rerun()
        else:
            st.warning("Please enter a name.")
    st.stop()  # Stop execution here until a user logs in

# MAIN APP LOGIC (For the logged-in user)
user = st.session_state.current_user
u_data = st.session_state.profiles[user]
rec = u_data['rec']

main_col, side_col = st.columns([2, 1])

with main_col:
    st.title(f"🔥 {user}'s Session")
    if st.button("⬅️ Switch Profile"):
        del st.session_state.current_user
        st.rerun()

    _, img_container, _ = st.columns([1, 2, 1])
    img_name = str(u_data['current_img'])
    img_path = os.path.join('females', img_name)

    with img_container:
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
            if u_data['match_traits']:
                st.success(f"✨ BEST MATCH (Score: {u_data['match_score']:.2%})")
                st.write("**Matched Traits:** " + ", ".join(u_data['match_traits']))
        else:
            st.error(f"Image {img_name} not found.")

    st.write("---")
    c1, c2, c3 = st.columns(3)


    def refresh_view(fname, fidx, traits=None, score=0):
        u_data['current_img'] = fname
        u_data['current_idx'] = fidx
        u_data['match_traits'] = traits
        u_data['match_score'] = score


    with c1:
        if st.button("❌ DISLIKE", use_container_width=True):
            rec.update_profile(u_data['current_idx'], False)
            refresh_view(*rec.get_next())
            st.rerun()
    with c2:
        if st.button("❤️ LIKE", use_container_width=True, type="primary"):
            rec.update_profile(u_data['current_idx'], True)
            refresh_view(*rec.get_next())
            st.rerun()
    with c3:
        if st.button("✨ MY MATCH", use_container_width=True):
            fname, fidx, traits, score = rec.find_best_match()
            refresh_view(fname, fidx, traits, score)
            st.rerun()

with side_col:
    st.header("Profile Opinion")
    st.metric("Likes for this User", rec.liked_count)

    n_attr = len(rec.attr_names)
    traits_df = pd.DataFrame({
        'Trait': rec.attr_names,
        'Score': rec.user_profile[:n_attr]
    }).sort_values(by='Score', ascending=False)

    st.dataframe(traits_df, height=500, hide_index=True)