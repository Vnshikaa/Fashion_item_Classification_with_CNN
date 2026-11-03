from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

# ---------------- SETTINGS (edit these) ----------------

APP_NAME = "StyleSense"
TAGLINE = "Instant clothing recognition with deep learning"
TEST_ACCURACY = "91.0%"          # <- replace with YOUR notebook's final test accuracy
MODEL_PATH = Path(__file__).parent / "fashion_mnist_cnn.keras"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title=APP_NAME, page_icon="👕", layout="centered")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp { background: #0B1215; color: #F1F5F9; }
.block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stHeader"], [data-testid="stToolbar"] { background: #0B1215; }

.hero { text-align: center; padding: 8px 0 22px 0; }
.hero h1 { color: #F1F5F9; font-size: 42px; font-weight: 800; letter-spacing: -1px; margin-bottom: 6px; }
.hero .tag { color: #5EEAD4; font-size: 18px; font-weight: 600; margin: 0 0 6px 0; }
.hero .sub { color: #94A3B8; font-size: 14px; margin: 0; }

.intro { background: linear-gradient(135deg, #10201F, #132A2A); border: 1px solid #1F5F5B;
         border-radius: 16px; padding: 22px; margin: 10px 0 26px 0; }
.intro h3 { color: #FFFFFF; margin-top: 0; }
.intro p { color: #CBD5E1; line-height: 1.7; margin-bottom: 0; }

.info-card { background: #111C1F; border: 1px solid #1E3A3D; border-radius: 12px;
             padding: 16px 8px; text-align: center; min-height: 100px; }
.info-card .value { color: #5EEAD4; font-size: 22px; font-weight: 800; }
.info-card .label { color: #94A3B8; font-size: 12px; margin-top: 5px; }

[data-testid="stFileUploader"] { background: #111C1F; border: 1px dashed #2DD4BF;
                                 border-radius: 14px; padding: 14px; }
[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploader"] button { background: #134E4A !important; color: #CCFBF1 !important;
                                        border: 1px solid #2DD4BF !important; border-radius: 9px !important; }
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small { color: #CBD5E1 !important; }

[data-testid="stImage"] img { border-radius: 14px; border: 1px solid #1E3A3D; }

.result-card { background: linear-gradient(135deg, #10201F, #132A2A); border: 1px solid #2DD4BF;
               border-radius: 16px; padding: 26px; text-align: center; margin-top: 8px; }
.result-label { color: #94A3B8; font-size: 13px; letter-spacing: 2px; font-weight: 600; }
.result-name { color: #5EEAD4; font-size: 32px; font-weight: 800; margin: 10px 0; }
.result-score { color: #FFFFFF; font-size: 34px; font-weight: 800; }
.result-caption { color: #94A3B8; font-size: 13px; }

[data-testid="stAlert"] { border-radius: 10px; }
.stProgress > div > div > div { background: #2DD4BF !important; }
hr { border-color: #1E3A3D; }
.footer { text-align: center; color: #64748B; font-size: 13px; line-height: 1.8; padding: 18px 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL + PREPROCESSING ----------------


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    """Turn any uploaded photo into a 28x28 grayscale array like Fashion-MNIST."""
    img = ImageOps.exif_transpose(img)  # fixes rotated phone photos

    # Put transparent PNGs on a white background
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    img = img.convert("L")

    # Center-crop to a square so the item is not stretched
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))

    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0

    # Fashion-MNIST items are light on a dark background.
    # Normal photos have a bright background, so invert them.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 0.5:
        arr = 1.0 - arr
    return arr


model = load_model()

# ---------------- HERO + INTRO ----------------

st.markdown(f"""
<div class="hero">
<h1>👕 {APP_NAME}</h1>
<p class="tag">{TAGLINE}</p>
<p class="sub">Convolutional Neural Network trained on the Fashion-MNIST dataset</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="intro">
<h3>How it works</h3>
<p>Upload a photo of a single clothing item or accessory. The image is converted to a
28x28 grayscale picture and a CNN predicts which of the 10 fashion categories it belongs to.
For best results use one centered item on a plain background.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------

st.markdown("### 🧠 Model Information")
cols = st.columns(4)
info = [("CNN", "Model Architecture"), (TEST_ACCURACY, "Test Accuracy"),
        ("10", "Fashion Classes"), ("28×28", "Input Image Size")]
for col, (value, label) in zip(cols, info):
    with col:
        st.markdown(f"""
<div class="info-card">
<div class="value">{value}</div>
<div class="label">{label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- UPLOAD + PREDICT ----------------

st.markdown("### 📸 Upload Your Image")
st.caption("Supported formats: PNG, JPG, JPEG, WEBP")
uploaded = st.file_uploader("Choose a clothing image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("Please upload a clothing image to see the prediction.")
else:
    original = Image.open(uploaded)
    processed = preprocess(original)

    with st.spinner("Analyzing your image..."):
        probs = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]

    top3 = np.argsort(probs)[::-1][:3]
    best = int(top3[0])
    confidence = float(probs[best]) * 100

    c1, c2 = st.columns(2)
    with c1:
        st.image(original, caption="Your image", width=240)
    with c2:
        st.image(processed, clamp=True, caption="What the model sees (28x28)", width=240)

    st.markdown("### 🎯 Prediction")
    if confidence < 70:
        st.warning("Low confidence. The model is not very sure about this image. "
                   "Try a clearer photo of a single item on a plain background.")

    st.markdown(f"""
<div class="result-card">
<div class="result-label">PREDICTED CATEGORY</div>
<div class="result-name">{CLASS_NAMES[best]}</div>
<div class="result-score">{confidence:.1f}%</div>
<div class="result-caption">Model confidence</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🏆 Top 3 Guesses")
    for i in top3:
        st.markdown(f"**{CLASS_NAMES[i]}** — {probs[i] * 100:.1f}%")
        st.progress(float(probs[i]))

# ---------------- FOOTER ----------------

st.divider()
st.markdown(f"""
<div class="footer">
<strong>{APP_NAME}</strong><br>
Deep Learning Project | Fashion-MNIST | CNN<br>
Built with TensorFlow, Keras and Streamlit
</div>
""", unsafe_allow_html=True)
