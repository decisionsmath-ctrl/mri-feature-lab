"""
app.py - Streamlit front end for MRI Feature Lab.

Run locally with:   streamlit run app.py
(NOT `python app.py`)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="MRI Feature Lab", page_icon="🧠", layout="wide")

MODEL_PATH = "models/mobilenet_mri.keras"

DISCLAIMER = (
    "**NOT FOR CLINICAL USE.** This is an educational demonstration. "
    "It is not a medical device and must never be used to diagnose, "
    "treat, or guide clinical decisions."
)


# --------------------------------------------------------------------------
# TensorFlow is imported lazily so the page renders immediately instead of
# freezing for 10-60 s on first load (this is what looked like "stuck").
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading TensorFlow and the model...")
def get_trained_model():
    import model_utils
    return model_utils.load_trained_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading ImageNet MobileNetV2 backbone...")
def get_backbone_model():
    import model_utils
    return model_utils.build_mobilenet_model()


def plot_feature_maps(feature_maps, max_channels=8):
    n_rows = len(feature_maps)
    fig, axes = plt.subplots(n_rows, max_channels,
                             figsize=(2 * max_channels, 2.2 * n_rows),
                             squeeze=False)
    for r, (name, act) in enumerate(feature_maps.items()):
        act = act[0]                       # drop batch axis -> (H, W, C)
        for c in range(max_channels):
            ax = axes[r][c]
            ax.axis("off")
            if c < act.shape[-1]:
                ax.imshow(act[:, :, c], cmap="viridis")
            if c == 0:
                ax.set_title(f"{name}\n{act.shape[0]}x{act.shape[1]}x{act.shape[2]}",
                             fontsize=8, loc="left")
    fig.tight_layout()
    return fig


def get_gemini_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def main():
    st.title("🧠 MRI Feature Lab")
    st.error(DISCLAIMER)

    uploaded = st.file_uploader("Upload a brain MRI image",
                                type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image.")
        return

    left, right = st.columns(2)
    left.image(image, caption="Uploaded image", use_container_width=True)

    import model_utils
    processed = model_utils.preprocess_image(image)

    # ---- Prediction (only with real trained weights) ---------------------
    model = get_trained_model()
    result = None
    with right:
        st.subheader("Classification")
        if model is None:
            st.warning(
                f"No trained model found at `{MODEL_PATH}`. "
                "No prediction is shown (nothing is fabricated). "
                "Train one with `python train.py` and add the file."
            )
        else:
            label, confidence, scores = model_utils.predict_mri(model, processed)
            result = (label, confidence, scores)
            st.metric("Model output", label, f"score {confidence:.3f}")
            st.bar_chart(scores)
            st.caption("Scores are model outputs, not calibrated "
                       "clinical probabilities.")

    # ---- Feature maps ----------------------------------------------------
    st.subheader("Intermediate feature maps")
    layer_count = st.slider("Number of layers", 2, 6, 4)
    if model is not None:
        source, note = model, "Feature maps from your trained model."
    else:
        source = get_backbone_model()
        note = ("Feature maps from the **ImageNet-pretrained backbone** "
                "(not trained on MRI).")
    st.caption(note)
    maps = model_utils.extract_feature_maps(source, processed, layer_count)
    if maps:
        fig = plot_feature_maps(maps)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No feature maps available.")

    # ---- Optional Gemini explanation ------------------------------------
    st.subheader("Optional: Gemini educational explanation")
    if result is None:
        st.caption("Available after a trained model produces a result.")
    else:
        st.caption("This sends the uploaded image to Google's Gemini API.")
        if st.button("Explain with Gemini"):
            from gemini_explainer import generate_mri_explanation
            with st.spinner("Asking Gemini..."):
                text = generate_mri_explanation(
                    image, result[0], result[1], result[2],
                    api_key=get_gemini_key(),
                )
            st.markdown(text)


main()
