import streamlit as st
import requests
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(page_title="number prediction", layout="centered")

st.markdown("""
    <style>
        body, .stApp {
            background-color: #111111;
            color: #ffffff;
        }
        h1 {
            color: #ffffff;
            font-family: 'Courier New', monospace;
            letter-spacing: 4px;
            text-align: center;
            font-size: 1.4rem;
            margin-bottom: 0.2rem;
        }
        .sub {
            text-align: center;
            color: #555;
            font-family: monospace;
            font-size: 0.75rem;
            margin-bottom: 1.2rem;
            letter-spacing: 2px;
        }
        .stButton > button {
            background-color: #1a1a1a;
            color: #ffffff;
            border: 1px solid #444;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            letter-spacing: 2px;
            font-size: 0.75rem;
            padding: 0.4rem 1rem;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #ffffff;
            color: #000000;
            border-color: #ffffff;
        }
        canvas {
            border: 1px solid #333 !important;
            border-radius: 4px;
            image-rendering: pixelated;
        }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>number prediction</h1>", unsafe_allow_html=True)
st.markdown('<p class="sub">BLACK CANVAS · WHITE PEN · PIXEL ART</p>', unsafe_allow_html=True)

# Display size (scaled up so it's usable)
DISPLAY_SIZE = 560   # 560 / 28 = 20x zoom
PIXEL_SIZE   = 28

# Controls
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    clear = st.button("CLEAR")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if clear:
    st.session_state.canvas_key += 1

# Draw on a 560×560 canvas, then downsample to 28×28
canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=28,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=DISPLAY_SIZE,
    width=DISPLAY_SIZE,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
    display_toolbar=False,
)


# Preview + download
if canvas_result.image_data is not None:
    img_array = canvas_result.image_data.astype(np.uint8)  # RGBA from canvas
 
    # --- Pipeline: RGBA canvas → grayscale → RGB (28×28×3, white on black) ---
    # Extract intensity using alpha mask
    r = img_array[:, :, 0].astype(np.float32)
    a = img_array[:, :, 3].astype(np.float32) / 255.0
    gray = (r * a).astype(np.uint8)           # (H, W) single channel
 
    img_gray = Image.fromarray(gray, mode="L")
 
    # Downsample to 28×28
    img_28_gray = img_gray.resize((PIXEL_SIZE, PIXEL_SIZE), Image.LANCZOS)
 
    # Convert to RGB (28×28×3) — R=G=B, white on black, matches reference format
    img_28_rgb = img_28_gray.convert("RGB")
    arr = np.array(img_28_rgb)  # shape: (28, 28, 3)
 
    st.markdown("---")
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("**28 × 28 Output**")
        img_display = img_28_rgb.resize((280, 280), Image.NEAREST)
        st.image(img_display)
        st.caption(f"Mode: {img_28_rgb.mode} | Shape: {arr.shape} | dtype: {arr.dtype}")
 
    with col_b:
        st.markdown("**Pixel values — R channel (0–255)**")
        st.dataframe(arr[:, :, 0], height=290)
 
    # Save as RGB PNG (28×28×3) — same format as reference image
    buf = io.BytesIO()
    img_28_rgb.save(buf, format="PNG")
    if st.button("Predict Digit"):
        buf.seek(0)
        byte_data = buf.read()
        files = {'image': ('canvas.png', byte_data, 'image/png')}
        response = requests.post("https://number-pre.onrender.com", files=files)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Digit: {result['predicted_digit']}    \nConfidence: {result['confidence']*10:.2f}%")
        else:
            st.error("Prediction failed. Please try again.")
