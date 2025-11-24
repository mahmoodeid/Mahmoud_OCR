import time
from typing import Optional

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# --- OPTIONAL: on Windows, point to tesseract.exe ---
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- Streamlit page config -----------------
st.set_page_config(
    page_title="Live OCR Camera",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Live Camera OCR Viewer")

st.markdown(
    """
This app opens your webcam, runs OCR on the live feed,  
and continuously updates the recognized text below the video.
"""
)

# ----------------- Session state init -----------------
if "cap" not in st.session_state:
    st.session_state.cap: Optional[cv2.VideoCapture] = None

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# ----------------- Sidebar controls -----------------
st.sidebar.header("Settings")

fps = st.sidebar.slider("Target FPS", min_value=1, max_value=15, value=5)
delay = 1.0 / fps

ocr_mode = st.sidebar.selectbox(
    "OCR mode",
    ["Digits only (0-9)", "Any text"],
    index=0,
)

roi_help = st.sidebar.checkbox(
    "Enable ROI (crop center region before OCR)",
    value=False,
    help="If enabled, OCR is only run on a central crop of the frame.",
)

roi_fraction = st.sidebar.slider(
    "ROI size (fraction of shorter side)",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown("Use **Start camera** to begin. Use **Stop camera** to close it.")


# ----------------- Button handlers -----------------
col_start, col_stop = st.columns(2)

with col_start:
    if st.button("▶️ Start camera", use_container_width=True):
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Could not open camera. Check your webcam and permissions.")
                st.session_state.cap = None

with col_stop:
    if st.button("⏹ Stop camera", use_container_width=True):
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.last_text = ""
            st.experimental_rerun()  # refresh UI after stopping


# ----------------- Placeholders for dynamic content -----------------
video_col, text_col = st.columns([2, 1])

with video_col:
    st.subheader("Camera Feed")
    video_placeholder = st.empty()

with text_col:
    st.subheader("Recognized Text")
    text_placeholder = st.empty()
    debug_placeholder = st.empty()


def preprocess_for_ocr(rgb_img: np.ndarray, enable_roi: bool) -> Image.Image:
    """
    Convert RGB frame to a PIL image suitable for OCR.
    Optionally crop center ROI.
    """
    h, w, _ = rgb_img.shape
    img = rgb_img

    if enable_roi:
        short_side = min(h, w)
        roi_size = int(short_side * roi_fraction)
        cx, cy = w // 2, h // 2
        x1 = max(cx - roi_size // 2, 0)
        x2 = min(cx + roi_size // 2, w)
        y1 = max(cy - roi_size // 2, 0)
        y2 = min(cy + roi_size // 2, h)
        img = img[y1:y2, x1:x2]

    # Convert to grayscale and apply simple thresholding to help OCR
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(thresh)
    return pil_img


def run_ocr(pil_img: Image.Image, mode: str) -> str:
    """
    Run Tesseract OCR on the image with basic configs.
    """
    if mode == "Digits only (0-9)":
        config = r"--psm 7 -c tessedit_char_whitelist=0123456789"
    else:
        # generic text; you can customize psm / language etc.
        config = r"--psm 6"

    text = pytesseract.image_to_string(pil_img, config=config)
    return text.strip()


# ----------------- Main loop: one frame per run -----------------
if st.session_state.cap is not None:
    ret, frame = st.session_state.cap.read()

    if not ret:
        st.error("Failed to read from camera.")
        st.session_state.cap.release()
        st.session_state.cap = None
    else:
        # Convert BGR (OpenCV) to RGB (Streamlit / PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show live frame
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # OCR processing
        pil_for_ocr = preprocess_for_ocr(frame_rgb, enable_roi=roi_help)
        ocr_text = run_ocr(pil_for_ocr, ocr_mode)

        st.session_state.last_text = ocr_text

        # Update text display
        if ocr_text:
            text_placeholder.markdown(f"**Latest OCR result:**  \n`{ocr_text}`")
        else:
            text_placeholder.markdown("*(No text recognized yet)*")

        # Optional: show debug info
        debug_placeholder.caption("OCR runs continuously while the camera is active.")

        # Small delay to control FPS, then rerun the script
        time.sleep(delay)
        st.experimental_rerun()
else:
    # When camera is off, show last known text (if any)
    if st.session_state.last_text:
        text_placeholder.markdown(f"**Last OCR result (camera off):**  \n`{st.session_state.last_text}`")
    else:
        text_placeholder.markdown("Camera is stopped. Start it to see live OCR.")
