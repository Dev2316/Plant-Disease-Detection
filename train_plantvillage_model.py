import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf


DEFAULT_DATASET_DIR = Path("/Users/devpatel/Downloads/Datasets/plantvillage dataset/plantvillage_split")
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "saved_models" / "1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multi-class PlantVillage model and export it for the LeafScan Flask app."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from a saved Keras checkpoint (.keras).",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Apply inverse-frequency class weights during training.",
    )
    return parser.parse_args()


def count_images_per_class(split_dir: Path) -> Counter:
    counts = Counter()
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(1 for file_path in class_dir.iterdir() if file_path.is_file())
    return counts


def build_class_weights(class_names: list[str], train_counts: Counter) -> dict[int, float]:
    total_images = sum(train_counts.values())
    num_classes = len(class_names)
    return {
        idx: total_images / (num_classes * train_counts[class_name])
        for idx, class_name in enumerate(class_names)
        if train_counts[class_name] > 0
    }


def normalize_dataset(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def prepare_datasets(dataset_dir: Path, image_size: int, batch_size: int, seed: int):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected train/val/test folders under "
            f"{dataset_dir}, but one or more splits are missing."
        )

    common_kwargs = {
        "image_size": (image_size, image_size),
        "batch_size": batch_size,
        "label_mode": "int",
        "seed": seed,
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        **common_kwargs,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=False,
        **common_kwargs,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        **common_kwargs,
    )

    class_names = list(train_ds.class_names)
    train_counts = count_images_per_class(train_dir)

    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.map(normalize_dataset, num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    val_ds = (
        val_ds.map(normalize_dataset, num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    test_ds = (
        test_ds.map(normalize_dataset, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    class_weights = build_class_weights(class_names, train_counts)
    return train_ds, val_ds, test_ds, class_names, train_counts, class_weights


def build_model(image_size: int, num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="leaf_image")
    x = data_augmentation(inputs)

    for filters, dropout_rate in [(24, 0.05), (48, 0.10), (96, 0.15), (160, 0.20)]:
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.SeparableConv2D(224, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(192, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="leafscan_plantvillage")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_saved_model(model: tf.keras.Model, export_dir: Path):
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        tf.saved_model.save(model, str(export_dir))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main():
    args = parse_args()

    tf.random.set_seed(args.seed)

    train_ds, val_ds, test_ds, class_names, train_counts, class_weights = prepare_datasets(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.resume_from:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from}")
        model = tf.keras.models.load_model(args.resume_from)
    else:
        model = build_model(image_size=args.image_size, num_classes=len(class_names))
    checkpoint_path = args.export_dir.parent / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print(f"[INFO] Training on dataset: {args.dataset_dir}")
    print(f"[INFO] Classes: {len(class_names)}")
    print(f"[INFO] Export directory: {args.export_dir}")

    fit_kwargs = {}
    if args.use_class_weights:
        fit_kwargs["class_weight"] = class_weights

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        **fit_kwargs,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    export_saved_model(model, args.export_dir)

    metadata = {
        "model_name": "leafscan_plantvillage",
        "dataset": "PlantVillage",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_counts": dict(train_counts),
        "test_metrics": {
            "loss": round(float(test_loss), 6),
            "accuracy": round(float(test_accuracy), 6),
        },
    }
    history_payload = {
        metric_name: [round(float(value), 6) for value in values]
        for metric_name, values in history.history.items()
    }

    write_json(args.export_dir / "model_metadata.json", metadata)
    write_json(args.export_dir / "class_names.json", class_names)
    write_json(args.export_dir / "training_history.json", history_payload)

    reloaded = tf.saved_model.load(str(args.export_dir))
    signatures = list(reloaded.signatures.keys())

    print("[INFO] Training complete.")
    print(f"[INFO] Test accuracy: {test_accuracy:.4f}")
    print(f"[INFO] SavedModel signatures: {signatures}")
    print(f"[INFO] Metadata written to: {args.export_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
