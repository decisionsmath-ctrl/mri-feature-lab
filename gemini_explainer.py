
"""
gemini_explainer.py
Educational explanations for MRI classification results using Gemini.
"""

import os
from PIL import Image
from google import genai
from google.genai import types

# gemini-2.5-* models are being retired (Oct 16, 2026) and return 404.
# Override without editing code:  set GEMINI_MODEL in env or Streamlit secrets.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

CLINICAL_DISCLAIMER = """
**Educational use only:** This experimental model is not a medical
diagnostic tool. Its predictions and confidence scores are not a diagnosis.
Consult a qualified radiologist or medical professional for clinical
interpretation.
"""


def generate_mri_explanation(
    pil_image: Image.Image,
    predicted_class: str,
    confidence: float,
    class_scores: dict,
    api_key: str = None,
    model_name: str = None,
) -> str:
    """Generate a cautious, educational explanation of model output."""

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    resolved_model = (
        model_name or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    )

    if not resolved_key:
        return (
            "Gemini is unavailable because no API key is configured. "
            "The MRI classifier can operate separately if its trained "
            "model is available."
        )

    prompt = f"""
You are an educational assistant explaining a computer-vision experiment.

A MobileNetV2 model classified an uploaded brain MRI image.

Model output:
- Predicted label: {predicted_class}
- Model confidence score: {confidence:.4f}
- Class scores: {class_scores}

Explain:
1. What a CNN learns in general when classifying images.
2. Why confidence scores do not establish diagnostic certainty.
3. How image quality, dataset bias, and distribution differences can
   affect predictions.
4. Why professional radiological assessment is necessary.

Important limitations:
- Do not claim that this classifier has detected or localized a tumor.
- Do not assert that a specific anatomical abnormality is present.
- The class scores are model outputs, not calibrated clinical probabilities.
- Do not provide a diagnosis, treatment recommendation, or prognosis.
- Clearly distinguish general CNN concepts from facts established by
  this particular model.
- Keep the explanation under 200 words.
"""

    try:
        client = genai.Client(api_key=resolved_key)

        response = client.models.generate_content(
            model=resolved_model,
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                # Generous cap: newer models may spend tokens on hidden
                # reasoning, which would otherwise leave the answer empty.
                max_output_tokens=2048,
            ),
        )

        explanation = getattr(response, "text", None)

        if not explanation or not explanation.strip():
            explanation = (
                "Gemini did not return a usable explanation. "
                "Please try again later."
            )

        return f"{explanation.strip()}\n\n---\n{CLINICAL_DISCLAIMER}"

    except Exception as exc:
        # Show a short, key-free reason so the problem can be diagnosed.
        reason = f"{type(exc).__name__}: {exc}".replace(resolved_key, "***")
        return (
            "Unable to generate an explanation.\n\n"
            f"`{resolved_model}` -> `{reason[:300]}`\n\n"
            "Common causes: 404 = model name retired (set GEMINI_MODEL), "
            "400/403 = bad or restricted API key, 429 = quota exceeded."
            f"\n\n---\n{CLINICAL_DISCLAIMER}"
        )
