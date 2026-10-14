"""
test_model_utils.py - Pytest suite verifying model generation,
image preprocessing pipeline, prediction logic, and failure handling.
"""

import os
import pytest
import numpy as np
from PIL import Image
import tensorflow as tf

from model_utils import (
    build_mobilenet_model,
    preprocess_image,
    load_trained_model,
    predict_mri,
    extract_feature_maps,
    IMG_SIZE,
    CLASS_NAMES
)

@pytest.fixture
def dummy_image():
    """Generates a dummy 200x200 RGB image for testing."""
    array = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")

@pytest.fixture
def compiled_model():
    """Instantiates an un-trained MobileNetV2 model fixture."""
    return build_mobilenet_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=2)

def test_preprocess_image(dummy_image):
    """Verifies image reshaping and array formatting."""
    processed = preprocess_image(dummy_image, target_size=IMG_SIZE)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)
    assert processed.dtype == np.float32

def test_build_mobilenet_model(compiled_model):
    """Verifies Keras model construction and output shape."""
    assert compiled_model is not None
    assert len(compiled_model.output_shape) == 2
    assert compiled_model.output_shape[-1] == len(CLASS_NAMES)

def test_absent_weights_handling():
    """Ensures missing weights file returns None without throwing unhandled exceptions."""
    result = load_trained_model("non_existent_path.keras")
    assert result is None

def test_prediction_output(compiled_model, dummy_image):
    """Verifies output format and confidence score ranges."""
    processed = preprocess_image(dummy_image, target_size=IMG_SIZE)
    pred_class, confidence, scores = predict_mri(compiled_model, processed)
    
    assert pred_class in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0
    assert len(scores) == len(CLASS_NAMES)

def test_extract_feature_maps(compiled_model, dummy_image):
    """Verifies feature map extraction returns activation tensors."""
    processed = preprocess_image(dummy_image, target_size=IMG_SIZE)
    fmaps = extract_feature_maps(compiled_model, processed, layer_count=2)
    
    assert isinstance(fmaps, dict)
    assert len(fmaps) <= 2
    for layer_name, tensor in fmaps.items():
        assert tensor.ndim == 4
