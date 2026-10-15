"""
model_utils.py - MobileNetV2 utilities for MRI image classification.

Pixel convention used everywhere in this project:
    * preprocess_image() returns pixels in the [0, 255] range.
    * The model itself contains a Rescaling layer that converts
      [0, 255] -> [-1, 1], which is what MobileNetV2 expects.
"""

import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (128, 128)
CLASS_NAMES = ["no_tumor", "tumor"]
DEFAULT_MODEL_PATH = "models/mobilenet_mri.keras"


def build_mobilenet_model(
    input_shape=(128, 128, 3),
    num_classes=2,
    trainable_base=False,
    weights="imagenet",
):
    """Build a binary MRI classifier using MobileNetV2.

    weights: "imagenet" (default) or None. Tests use None so that no
    download is needed.
    """

    if num_classes != len(CLASS_NAMES):
        raise ValueError("num_classes must be 2 for this classifier.")

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = trainable_base

    inputs = layers.Input(shape=input_shape, name="image")

    # Convert pixels from [0, 255] to [-1, 1], as MobileNetV2 expects.
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_MRI_Classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """Load a trained model, or return None if unavailable."""

    if not os.path.isfile(model_path):
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except (ValueError, OSError, TypeError) as exc:
        print(f"Could not load model: {exc}")
        return None


def preprocess_image(pil_image, target_size=IMG_SIZE):
    """Return a batch of RGB pixels in the [0, 255] range."""

    if not isinstance(pil_image, Image.Image):
        raise TypeError("Expected a PIL Image.")

    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(target_size)

    array = np.asarray(pil_image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_mri(model, processed_image_array):
    """Return predicted label, confidence, and class probabilities."""

    if model is None:
        raise ValueError("A trained model must be loaded before prediction.")

    predictions = np.asarray(
        model.predict(processed_image_array, verbose=0)[0]
    )

    if predictions.shape != (len(CLASS_NAMES),):
        raise ValueError("Model output does not match the expected two classes.")

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model returned non-finite prediction values.")

    predicted_idx = int(np.argmax(predictions))
    scores = {
        name: float(prob)
        for name, prob in zip(CLASS_NAMES, predictions)
    }

    return CLASS_NAMES[predicted_idx], float(predictions[predicted_idx]), scores


def extract_feature_maps(model, processed_image_array, layer_count=4):
    """Extract selected convolutional activations for visualization.

    Works with a full classifier from build_mobilenet_model() or with a
    bare MobileNetV2 backbone. Returns {layer_name: activation_array}.
    """

    if model is None or layer_count < 1:
        return {}

    base_model = next(
        (
            layer for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "mobilenet" in layer.name.lower()
        ),
        None,
    )

    if base_model is None:
        return {}

    candidates = [
        layer for layer in base_model.layers
        if isinstance(
            layer,
            (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D),
        )
    ]

    if not candidates:
        return {}

    indices = np.linspace(
        0,
        len(candidates) - 1,
        num=min(layer_count, len(candidates)),
        dtype=int,
    )
    selected = [candidates[i] for i in sorted(set(indices.tolist()))]

    # One multi-output extractor: a single forward pass for all layers.
    extractor = models.Model(
        inputs=base_model.input,
        outputs=[layer.output for layer in selected],
    )

    # The backbone expects [-1, 1]; preprocess_image() gives [0, 255].
    normalized = np.asarray(processed_image_array, dtype=np.float32) / 127.5 - 1.0

    activations = extractor.predict(normalized, verbose=0)
    if not isinstance(activations, (list, tuple)):
        activations = [activations]

    return {layer.name: act for layer, act in zip(selected, activations)}
