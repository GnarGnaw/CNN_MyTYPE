import streamlit as st
from PIL import Image
import os
import pandas as pd
from engine import Recommender

st.set_page_config(page_title="Celeb-Match", layout="wide")

if 'rec' not in st.session_state:
    st.session_state.rec = Recommender('attr.txt', 'landmarks.txt')
    fname, fidx = st.session_state.rec.get_next()
    st.session_state.current_img = fname
    st.session_state.current_idx = fidx
    st.session_state.match_mode = False

main_col, side_col = st.columns([2, 1])

with main_col:
    st.title("🔥 Celeb-Match")

    _, img_container, _ = st.columns([1, 2, 1])
    curr_img = str(st.session_state.current_img)
    img_path = os.path.join('females', curr_img)

    with img_container:
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
            if st.session_state.match_mode:
                score = st.session_state.get('match_score', 0)
                st.success(f"✨ GLOBAL BEST MATCH (Confidence: {score:.2%})")
                st.write("**Traits:** " + ", ".join(st.session_state.get('match_traits', [])))
        else:
            st.error(f"Image {curr_img} not found.")

    st.write("---")
    c1, c2, c3 = st.columns(3)


    def next_card():
        st.session_state.match_mode = False
        fname, fidx = st.session_state.rec.get_next()
        st.session_state.current_img = fname
        st.session_state.current_idx = fidx


    with c1:
        if st.button("❌ DISLIKE", use_container_width=True):
            st.session_state.rec.update_profile(st.session_state.current_idx, False)
            next_card()
            st.rerun()

    with c2:
        if st.button("❤️ LIKE", use_container_width=True, type="primary"):
            st.session_state.rec.update_profile(st.session_state.current_idx, True)
            next_card()
            st.rerun()

    with c3:
        if st.button("✨ FIND GLOBAL MATCH", use_container_width=True):
            fname, fidx, traits, score = st.session_state.rec.find_best_match()
            st.session_state.current_img = fname
            st.session_state.current_idx = fidx
            st.session_state.match_traits = traits
            st.session_state.match_score = score
            st.session_state.match_mode = True
            st.rerun()

with side_col:
    st.header("The AI's Opinion")
    n_attr = len(st.session_state.rec.attr_names)
    scores = st.session_state.rec.user_profile[:n_attr]
    traits_df = pd.DataFrame({'Trait': st.session_state.rec.attr_names, 'Score': scores}).sort_values(by='Score',
                                                                                                      ascending=False)
    st.metric("Total Likes", st.session_state.rec.liked_count)
    st.dataframe(traits_df, height=500, hide_index=True)