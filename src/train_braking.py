"""
Model A — Braking State CNN (Binary: normal_brakes vs worn_out_brakes)

Architecture target: < 100 KB after TFLite conversion.
Input shape: (40, 78, 1)  — 40 MFCCs × 78 time frames × 1 channel
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent.parent
DATA_PATH    = ROOT / "data" / "processed" / "braking.npz"
MODEL_PATH   = ROOT / "models" / "braking_model.h5"
CURVES_PATH  = ROOT / "results" / "training_curves" / "braking_curves.png"
CM_PATH      = ROOT / "results" / "confusion_matrices" / "braking_cm.png"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_shape: tuple = (40, 78, 1), n_classes: int = 2):
    """Lightweight CNN — estimated < 20 KB after int8 TFLite quantisation."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    l2 = regularizers.l2(1e-4)

    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="braking_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_splits(data_path: str | Path = DATA_PATH):
    """Load pre-processed braking splits and add channel dimension."""
    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"][..., np.newaxis].astype(np.float32)  # (N, 40, 78, 1)
    X_val   = d["X_val"][..., np.newaxis].astype(np.float32)
    X_test  = d["X_test"][..., np.newaxis].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    y_val   = d["y_val"].astype(np.int32)
    y_test  = d["y_test"].astype(np.int32)
    label_names = list(d["label_names"])
    return X_train, X_val, X_test, y_train, y_val, y_test, label_names


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path: Path = CURVES_PATH):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(history.history["loss"]) + 1)

    axes[0].plot(epochs, history.history["loss"],     label="train", linewidth=1.8)
    axes[0].plot(epochs, history.history["val_loss"], label="val",   linewidth=1.8)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1].plot(epochs, history.history["accuracy"],     label="train", linewidth=1.8)
    axes[1].plot(epochs, history.history["val_accuracy"], label="val",   linewidth=1.8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Model A — Braking CNN Training Curves", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_path: Path = CM_PATH):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalised (recall per class)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, linewidths=0.5, cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    fig.suptitle("Model A — Braking CNN Confusion Matrix (Test Set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    data_path:  str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    epochs:     int = 60,
    batch_size: int = 32,
    patience:   int = 15,
    seed:       int = 42,
):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, label_names = load_splits(data_path)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Classes: {label_names}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(input_shape=X_train.shape[1:], n_classes=len(label_names))
    model.summary()

    total_params = model.count_params()
    size_kb = total_params * 4 / 1024
    print(f"\nParam count : {total_params:,}")
    print(f"Float32 size: {size_kb:.1f} KB  (TFLite int8 will be ~4× smaller)\n")

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    _, val_acc  = model.evaluate(X_val,  y_val,  verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nVal  accuracy : {val_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_training_curves(history)
    plot_confusion_matrix(y_test, y_pred, label_names)

    print(f"\nModel saved -> {model_path}")
    return model, history, (val_acc, test_acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
