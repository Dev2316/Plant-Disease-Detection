import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf


DEFAULT_DATASET_DIR = Path("/Users/devpatel/Downloads/Datasets/plantvillage dataset/plantvillage_split")
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "saved_models" / "1"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "saved_models" / "transfer_best.keras"
MOBILENET_WEIGHTS_PATH = Path.home() / ".keras" / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a stronger color-only PlantVillage model with transfer learning."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_images_per_class(split_dir: Path) -> dict[str, int]:
    return {
        class_dir.name: sum(1 for item in class_dir.iterdir() if item.is_file())
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir())
    }


def load_split(dataset_dir: Path, split: str, image_size: int, batch_size: int, seed: int, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        dataset_dir / split,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        seed=seed,
        shuffle=shuffle,
    )


def prepare_datasets(dataset_dir: Path, image_size: int, batch_size: int, seed: int):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    for required_dir in [train_dir, val_dir, test_dir]:
        if not required_dir.exists():
            raise FileNotFoundError(f"Missing dataset split directory: {required_dir}")

    train_ds = load_split(dataset_dir, "train", image_size, batch_size, seed, shuffle=True)
    val_ds = load_split(dataset_dir, "val", image_size, batch_size, seed, shuffle=False)
    test_ds = load_split(dataset_dir, "test", image_size, batch_size, seed, shuffle=False)

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names, count_images_per_class(train_dir)


def build_model(image_size: int, num_classes: int):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="leaf_image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(image_size, image_size, 3),
    )
    if MOBILENET_WEIGHTS_PATH.exists():
        backbone.load_weights(MOBILENET_WEIGHTS_PATH)
    else:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="leafscan_plantvillage_transfer")
    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def merge_histories(*histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(round(float(v), 6) for v in values)
    return merged


def load_best_model(checkpoint_path: Path) -> tf.keras.Model:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return tf.keras.models.load_model(checkpoint_path)


def find_backbone_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
            return layer
    raise ValueError("Could not locate MobileNetV2 backbone inside the loaded model.")


def unfreeze_backbone(backbone: tf.keras.Model, unfreeze_layers: int):
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in backbone.layers[-unfreeze_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def export_saved_model(model: tf.keras.Model, export_dir: Path):
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(export_dir))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)

    train_ds, val_ds, test_ds, class_names, train_counts = prepare_datasets(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model, backbone = build_model(image_size=args.image_size, num_classes=len(class_names))
    checkpoint_path = args.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-6,
        ),
    ]

    print(f"[INFO] Dataset: {args.dataset_dir}")
    print(f"[INFO] Color-only classes: {len(class_names)}")
    print(f"[INFO] Image size: {args.image_size}")
    print(f"[INFO] Batch size: {args.batch_size}")

    compile_model(model, learning_rate=1e-3)
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if args.finetune_epochs > 0:
        best_model = load_best_model(checkpoint_path)
        backbone = find_backbone_model(best_model)
        unfreeze_backbone(backbone, unfreeze_layers=args.unfreeze_layers)
        compile_model(best_model, learning_rate=1e-5)
        finetune_history = best_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs + args.finetune_epochs,
            initial_epoch=args.head_epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        best_model = load_best_model(checkpoint_path)
        finetune_history = None

    best_model = load_best_model(checkpoint_path)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    export_saved_model(best_model, args.export_dir)

    histories = [head_history]
    if finetune_history is not None:
        histories.append(finetune_history)
    merged_history = merge_histories(*histories)
    metadata = {
        "model_name": "leafscan_plantvillage_transfer",
        "dataset": "PlantVillage color split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_value_range": "raw_0_255",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_counts": train_counts,
        "checkpoint_path": str(checkpoint_path),
        "test_metrics": {
            "loss": round(float(test_loss), 6),
            "accuracy": round(float(test_accuracy), 6),
        },
    }

    write_json(args.export_dir / "model_metadata.json", metadata)
    write_json(args.export_dir / "class_names.json", class_names)
    write_json(args.export_dir / "training_history.json", merged_history)

    print("[INFO] Transfer-learning training complete.")
    print(f"[INFO] Test accuracy: {test_accuracy:.4f}")
    print(f"[INFO] Exported model to: {args.export_dir}")


if __name__ == "__main__":
    main()
