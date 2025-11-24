import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image


# ---------------------- CONFIG ----------------------
# Uncomment and set this on Windows if needed:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MAX_CAMERA_INDEX = 5  # how many indices to probe for USB cams
DEFAULT_IP_URL = "http://192.168.1.15:8080/video"  # example for IP Webcam / DroidCam


# ---------------------- PAGE SETUP ----------------------
st.set_page_config(
    page_title="Live OCR Camera (USB or Mobile)",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Live OCR from USB or Mobile Camera")

st.markdown(
    """
This app can read text from:
- A **USB / built-in camera**, or  
- A **mobile phone camera over Wi-Fi** (IP camera URL)  

It runs OCR (Tesseract) continuously and shows the recognized text.
"""
)


# ---------------------- SESSION STATE INIT ----------------------
if "cap" not in st.session_state:
    st.session_state.cap: Optional[cv2.VideoCapture] = None

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

if "camera_list" not in st.session_state:
    # List of (index, label) for USB cameras
    st.session_state.camera_list: List[Tuple[int, str]] = []

if "current_source" not in st.session_state:
    # "usb" or "ip"
    st.session_state.current_source = None

if "current_usb_index" not in st.session_state:
    st.session_state.current_usb_index: Optional[int] = None

if "current_ip_url" not in st.session_state:
    st.session_state.current_ip_url: str = DEFAULT_IP_URL


# ---------------------- HELPER FUNCTIONS ----------------------
def scan_usb_cameras(max_index: int = MAX_CAMERA_INDEX) -> List[Tuple[int, str]]:
    """
    Probe camera indices [0..max_index-1] and return those that open successfully.
    """
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            # Try to read one frame to confirm
            ret, _ = cap.read()
            if ret:
                label = f"Camera {idx}"
                found.append((idx, label))
        if cap is not None:
            cap.release()
    return found


def release_camera():
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None


def preprocess_for_ocr(rgb_img: np.ndarray, roi_fraction: float) -> Image.Image:
    """
    Convert RGB frame to a PIL image suitable for OCR.
    Crop the center ROI with given fraction (0..1 of shorter side).
    """
    h, w, _ = rgb_img.shape
    img = rgb_img

    short_side = min(h, w)
    roi_size = int(short_side * roi_fraction)
    cx, cy = w // 2, h // 2

    x1 = max(cx - roi_size // 2, 0)
    x2 = min(cx + roi_size // 2, w)
    y1 = max(cy - roi_size // 2, 0)
    y2 = min(cy + roi_size // 2, h)

    img = img[y1:y2, x1:x2]

    # Grayscale + blur + threshold to help OCR
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(thresh)
    return pil_img


def run_ocr(pil_img: Image.Image, mode: str) -> str:
    if mode == "Digits only (0-9)":
        config = r"--psm 7 -c tessedit_char_whitelist=0123456789"
    else:
        config = r"--psm 6"
    text = pytesseract.image_to_string(pil_img, config=config)
    return text.strip()


# ---------------------- SIDEBAR: SOURCE SELECTION ----------------------
st.sidebar.header("Camera Source")

source_type = st.sidebar.radio(
    "Select camera source:",
    options=["Local / USB camera", "Mobile camera over Wi-Fi (IP camera)"],
    index=0,
)

ocr_mode = st.sidebar.selectbox(
    "OCR mode",
    ["Digits only (0-9)", "Any text"],
    index=0,
)

roi_fraction = st.sidebar.slider(
    "ROI size (fraction of shorter side)",
    min_value=0.2,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="OCR is run on a central crop of the frame.",
)

fps = st.sidebar.slider("Target FPS", min_value=1, max_value=15, value=5)
delay = 1.0 / fps

st.sidebar.markdown("---")


# ---------------------- SOURCE-SPECIFIC UI ----------------------
if source_type == "Local / USB camera":
    st.session_state.current_source = "usb"

    if st.sidebar.button("🔍 Scan for cameras"):
        cams = scan_usb_cameras()
        st.session_state.camera_list = cams
        if not cams:
            st.sidebar.error("No cameras detected.")
        else:
            st.sidebar.success(f"Detected {len(cams)} camera(s).")

    if st.session_state.camera_list:
        labels = [label for _, label in st.session_state.camera_list]
        indices = [idx for idx, _ in st.session_state.camera_list]

        selected_label = st.sidebar.selectbox(
            "Select USB camera",
            options=labels,
            index=0,
        )
        selected_idx = indices[labels.index(selected_label)]
        st.session_state.current_usb_index = selected_idx
    else:
        st.sidebar.info("Click **Scan for cameras** to populate the list.")

else:
    st.session_state.current_source = "ip"

    ip_url = st.sidebar.text_input(
        "IP camera URL (from mobile app)",
        value=st.session_state.current_ip_url or DEFAULT_IP_URL,
        help="Example for IP Webcam / DroidCam: http://PHONE_IP:PORT/video",
    )
    st.session_state.current_ip_url = ip_url


# ---------------------- START / STOP BUTTONS ----------------------
col_start, col_stop = st.columns(2)

with col_start:
    if st.button("▶️ Start", use_container_width=True):
        release_camera()  # reset any previous capture
        if st.session_state.current_source == "usb":
            if st.session_state.current_usb_index is None:
                st.error("No USB camera selected. Scan and choose one from the dropdown.")
            else:
                idx = st.session_state.current_usb_index
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    st.error(f"Failed to open USB camera {idx}.")
                else:
                    st.session_state.cap = cap
        else:  # IP camera
            url = st.session_state.current_ip_url.strip()
            if not url:
                st.error("Please enter a valid IP camera URL.")
            else:
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    st.error("Failed to open IP camera stream. Check URL and network.")
                else:
                    st.session_state.cap = cap

with col_stop:
    if st.button("⏹ Stop", use_container_width=True):
        release_camera()
        st.session_state.last_text = ""
        st.experimental_rerun()


# ---------------------- MAIN LAYOUT ----------------------
video_col, text_col = st.columns([2, 1])

with video_col:
    st.subheader("Camera Feed")
    video_placeholder = st.empty()

with text_col:
    st.subheader("Recognized Text")
    text_placeholder = st.empty()
    debug_placeholder = st.empty()


# ---------------------- MAIN LOOP: ONE FRAME PER RUN ----------------------
if st.session_state.cap is not None:
    ret, frame = st.session_state.cap.read()

    if not ret:
        st.error("Failed to read from camera. It may have been disconnected.")
        release_camera()
    else:
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show frame
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # OCR
        pil_for_ocr = preprocess_for_ocr(frame_rgb, roi_fraction=roi_fraction)
        ocr_text = run_ocr(pil_for_ocr, ocr_mode)

        st.session_state.last_text = ocr_text

        if ocr_text:
            text_placeholder.markdown(f"**Latest OCR result:**\n\n`{ocr_text}`")
        else:
            text_placeholder.markdown("*(No text recognized yet)*")

        debug_placeholder.caption(
            f"Source: {st.session_state.current_source.upper()} | OCR mode: {ocr_mode} | FPS≈{fps}"
        )

        time.sleep(delay)
        st.experimental_rerun()
else:
    # When camera is off
    if st.session_state.last_text:
        text_placeholder.markdown(
            f"**Last OCR result (camera stopped):**\n\n`{st.session_state.last_text}`"
        )
    else:
        text_placeholder.markdown("Camera is stopped. Choose a source and click **Start**.")
    debug_placeholder.caption("No active camera.")
