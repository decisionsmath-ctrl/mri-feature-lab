"""
train.py - Train the MobileNetV2 MRI classifier.

Expected folder layout (class folder names must match CLASS_NAMES):

    data/train/no_tumor/*.jpg   data/train/tumor/*.jpg
    data/val/no_tumor/*.jpg     data/val/tumor/*.jpg

Run:
    python train.py --epochs 10
    python train.py --epochs 10 --fine-tune-epochs 5
"""

import argparse
import os

import tensorflow as tf

from model_utils import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    IMG_SIZE,
    build_mobilenet_model,
)


def load_split(path, batch_size, shuffle):
    if not os.path.isdir(path):
        raise SystemExit(f"Folder not found: {path}")

    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",          # matches sparse_categorical_crossentropy
        class_names=CLASS_NAMES,   # fixes the order: no_tumor=0, tumor=1
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    ).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    train_ds = load_split(args.train_dir, args.batch_size, shuffle=True)
    val_ds = load_split(args.val_dir, args.batch_size, shuffle=False)

    model = build_mobilenet_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]

    print("Stage 1: training the classification head (backbone frozen)")
    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        print("Stage 2: fine-tuning the top of the backbone")
        base = next(l for l in model.layers if "mobilenet" in l.name.lower()
                    and isinstance(l, tf.keras.Model))
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds,
                  epochs=args.fine_tune_epochs, callbacks=callbacks)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
