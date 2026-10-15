# MRI Feature Lab: Deep Learning & Feature Visualization Framework

An end-to-end Python project using **Streamlit**, **TensorFlow/Keras (MobileNetV2)**, **Pillow**, and the **Google Gen AI SDK** for binary brain MRI analysis (`no_tumor` vs `tumor`).

---

## 🛑 Clinical & Medical Disclaimer
**NOT FOR CLINICAL USE:** This software is an educational and technical demonstration tool. It has not been approved by the FDA, CE, or any health authority. It must **NEVER** be used to diagnose, treat, or guide clinical decisions.

---

## 🌟 Key Features
- **MobileNetV2 transfer learning** for binary classification (`no_tumor` vs `tumor`).
- **No-fabrication guardrail:** if no trained model file exists, the app shows no prediction.
- **Intermediate activation visualizer:** renders feature maps from convolutional layers.
- **Optional Gemini assistant:** educational explanation via the `google-genai` SDK.
- **Pytest suite** for the image pipeline and model utilities.

---

## 📁 Project Structure
```
app.py                 Streamlit app
model_utils.py         Model, preprocessing, prediction, feature maps
train.py               Training script
gemini_explainer.py    Optional Gemini explanation
tests/                 Pytest suite
models/                Trained model goes here (mobilenet_mri.keras)
data/train, data/val   Images in no_tumor/ and tumor/ subfolders
```

## 🛠️ Installation & Setup

```bash
git clone https://github.com/your-username/mri-feature-lab.git
cd mri-feature-lab
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
```

Use **Python 3.10 – 3.12** (TensorFlow does not support every newest Python release).

### Train a model
```bash
python train.py --epochs 10
```
This writes `models/mobilenet_mri.keras`.

### Run the app
```bash
streamlit run app.py
```
Always use `streamlit run app.py`, not `python app.py`. The first load can take a while because TensorFlow is starting up.

### Gemini (optional)
Set `GEMINI_API_KEY` as an environment variable, or in `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key"
```
Never commit this file.

### Run tests
```bash
pytest
```

## ☁️ Deploying on Streamlit Community Cloud
1. Commit `models/mobilenet_mri.keras` to the repo (about 12 MB).
2. In **Advanced settings**, choose Python 3.11 or 3.12.
3. Add `GEMINI_API_KEY` under **Secrets** if you want the Gemini feature.
