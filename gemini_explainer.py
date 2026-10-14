```python
"""
gemini_explainer.py
Educational explanations for MRI classification results using Gemini.
"""

import os
from PIL import Image
from google import genai
from google.genai import types

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
) -> str:
    """Generate a cautious, educational explanation of model output."""

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")

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
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=350,
            ),
        )

        explanation = getattr(response, "text", None)

        if not explanation or not explanation.strip():
            explanation = (
                "Gemini did not return a usable explanation. "
                "Please try again later."
            )

        return f"{explanation.strip()}\n\n---\n{CLINICAL_DISCLAIMER}"

    except Exception:
        # Avoid displaying raw exception details in the user interface.
        return (
            "Unable to generate an explanation at this time. "
            "Check the API configuration and service availability. "
            "The classification result should be considered separately."
            f"\n\n---\n{CLINICAL_DISCLAIMER}"
        )
```
