

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Global configuration constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ["no_tumor", "tumor"]

def build_mobilenet_model(input_shape=(128, 128, 3), num_classes=2, trainable_base=False):
    """
    Constructs a transfer learning model using pre-trained MobileNetV2 base.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = trainable_base

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)  # Normalize pixel values [0, 1]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_MRI_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def load_trained_model(model_path="models/mobilenet_mri.keras"):
    """
    Safely loads trained model weights if present.
    Returns None if weights file is absent to prevent fabricated predictions.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def preprocess_image(pil_image, target_size=IMG_SIZE):
    """
    Preprocesses a PIL image for input into the MobileNetV2 pipeline.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    resized_img = pil_image.resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32)
    expanded_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    return expanded_array

def predict_mri(model, processed_image_array):
    """
    Executes inference using a trained model.
    Returns predicted class, confidence, and class probabilities dict.
    """
    if model is None:
        raise ValueError("Model object is None. Trained weights must be supplied.")
    
    predictions = model.predict(processed_image_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    
    scores = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, predictions)}
    return CLASS_NAMES[predicted_idx], confidence, scores

def extract_feature_maps(model, processed_image_array, layer_count=4):
    """
    Extracts intermediate activation maps from MobileNetV2 Conv2D/Bottleneck layers.
    """
    if model is None:
        return {}

    # Extract base MobileNetV2 layer
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_layer = layer
            break

    target_layers = []
    if base_layer is not None:
        # Pull candidate Conv2D / DepthwiseConv2D layers from MobileNetV2 base
        for layer in base_layer.layers:
            if "conv" in layer.name.lower() or "out" in layer.name.lower():
                target_layers.append(layer)
    else:
        for layer in model.layers:
            if "conv" in layer.name.lower() or "dense" in layer.name.lower():
                target_layers.append(layer)

    if not target_layers:
        return {}

    # Select representative intermediate layers spaced evenly
    step = max(1, len(target_layers) // layer_count)
    selected_layers = target_layers[::step][:layer_count]

    # Re-normalize array for intermediate layer extraction pipeline
    rescaled_input = processed_image_array / 255.0

    feature_maps = {}
    for layer in selected_layers:
        sub_model = models.Model(inputs=base_layer.inputs, outputs=layer.output)
        feature_map = sub_model.predict(rescaled_input, verbose=0)
        feature_maps[layer.name] = feature_map

    return feature_maps
