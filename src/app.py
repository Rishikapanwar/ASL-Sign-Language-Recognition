import base64

import streamlit as st
import cv2
import os
import time
import torch
from lstm import ASLLSTM
from feature_extraction import MediaPipe
from utils import index_to_label, data_dir,checkpoints_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLLSTM()
model_path = os.path.join(checkpoints_dir, 'asl_lstm_final1.pth')
model.load_state_dict(torch.load(model_path))
model.eval()
extractor = MediaPipe()

st.markdown(
    """
    <style>
    /* Set the background color of the whole page */
    .stApp {
        background-color: #E6E6FA;  /* Light purple / Lavender */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .main > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Sign Language Name Speller", layout="centered")
st.title("☝️Sign your name")

#session state
if "letter_idx" not in st.session_state:
    st.session_state.letter_idx =0
if "correct_letters" not in st.session_state:
    st.session_state.correct_letters =[]
if "done" not in st.session_state:
    st.session_state.done = False
if "started" not in st.session_state:
    st.session_state.started = False
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

#enter username:
if not st.session_state.started:
    user_name = st.text_input("Enter your Name (letters only)")
    if st.button("start signing"):
        st.session_state.started = True
        st.session_state.user_name = user_name.upper()
        st.rerun()

if st.session_state.started:
    user_name = st.session_state.user_name

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    prompt_placeholder = st.empty()
    sign_grid_placeholder = st.empty()
    # Add prediction smoothing
    consecutive_matches = 0
    REQUIRED_MATCHES = 2
    cap = cv2.VideoCapture(0)
    run = True


    while run and st.session_state.letter_idx < len(user_name):
        with sign_grid_placeholder.container():
            cols = st.columns(len(user_name))
            for idx, letter in enumerate(user_name):
                img_path = os.path.join(data_dir, f"{letter}.png")
                border_color = "green" if idx < len(st.session_state.correct_letters) else "gray"
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div style="border: 3px solid {border_color}; padding: 2px; display: inline-block;">
                            <img src="data:image/png;base64,{base64.b64encode(open(img_path, 'rb').read()).decode()}" 
                                 style="height: 60px; width: auto;">
                            <div style="text-align: center;">{letter}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
        if "pause_until" in st.session_state:
            if time.time() < st.session_state.pause_until:
                frame_rgb = cv2.cvtColor(st.session_state.pause_frame, cv2.COLOR_BGR2RGB)
                cv2.putText(frame_rgb, st.session_state.pause_label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                            (255, 255, 255), 2, cv2.LINE_AA)
                frame_placeholder.image(frame_rgb, channels="RGB", width=500)
                status_placeholder.success(st.session_state.pause_label)
                continue
            else:
                # End pause and move to next letter
                del st.session_state.pause_until
                del st.session_state.pause_frame
                del st.session_state.pause_label
                st.session_state.correct_letters.append(st.session_state.user_name[st.session_state.letter_idx])
                st.session_state.letter_idx += 1
                consecutive_matches = 0
                continue

        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break
        target_letter = st.session_state.user_name[st.session_state.letter_idx]
        prompt_placeholder.markdown(f"### 👉 Show the sign for letter: **{target_letter}**")
        coords, frame_with_landmarks = extractor.extract_from_frame(frame)
        label = 'No hand detected'
        match = False

        if coords:
            coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
            #make real time prediction with trained model
            with torch.no_grad():
                logits = model(coords_tensor)
                pred = torch.argmax(logits, dim=1).item()
                #map indices to letters
                predicted_label = index_to_label.get(pred, "No clear sign")

            if predicted_label == target_letter:
                consecutive_matches += 1

                if consecutive_matches >= REQUIRED_MATCHES:
                    label = f"Correctly signed: {predicted_label}"
                    # Store pause state
                    st.session_state.pause_until = time.time() + 1.5
                    st.session_state.pause_frame = frame_with_landmarks.copy()
                    st.session_state.pause_label = label
                    continue
            else:
                label = f'Signed: {predicted_label}, Try again'
                consecutive_matches = 0


        cv2.putText(frame_with_landmarks, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    (255, 255, 255) if match else (255, 255, 255), 2, cv2.LINE_AA)
        frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels='RGB', width=500)
        status_placeholder.markdown(f"### {label}")

    cap.release()
    frame_placeholder.empty()
    status_placeholder.success("🎉 You completed the word!")

    if st.button("Sign another word"):
        st.session_state.started = False
        st.session_state.user_name = ""
        st.session_state.letter_idx = 0
        st.session_state.correct_letters = []
        st.rerun()


