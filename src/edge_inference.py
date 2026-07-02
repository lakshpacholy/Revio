"""
TFLite conversion and edge inference helpers for Revio.

Converts Keras .h5 models to .tflite (dynamic-range int8 quantised) and
provides a thin wrapper for running inference with the TFLite interpreter
— this is what the Streamlit demo and any real edge deployment will use.
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_tflite(
    h5_path: str | Path,
    tflite_path: str | Path,
    representative_data: np.ndarray | None = None,
    quantize: bool = True,
) -> Path:
    """Convert a Keras .h5 model to .tflite.

    If `representative_data` is given, full int8 quantisation is applied
    (smallest size, requires a representative sample of training inputs).
    Otherwise falls back to dynamic-range quantisation (weights only).
    """
    h5_path = Path(h5_path)
    tflite_path = Path(tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        _configure_quantization(converter, representative_data)

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Converted {h5_path.name} -> {tflite_path.name}  ({size_kb:.1f} KB)")
    return tflite_path


def _configure_quantization(converter, representative_data):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_data is not None:
        def representative_dataset():
            for sample in representative_data[:100]:
                yield [sample[np.newaxis, ...].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        # Keep float32 in/out so the demo can pass raw MFCCs directly
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TFLiteClassifier:
    """Thin wrapper around tf.lite.Interpreter for single-sample inference."""

    def __init__(self, tflite_path: str | Path, label_names: list[str]):
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.label_names = label_names

    def predict(self, mfcc: np.ndarray) -> dict:
        """Run inference on a single MFCC array of shape (40, 78) or (40, 78, 1).

        Returns dict with fault_label, confidence, all_probabilities, latency_ms.
        """
        if mfcc.ndim == 2:
            mfcc = mfcc[..., np.newaxis]
        x = mfcc[np.newaxis, ...].astype(self.input_details[0]["dtype"])

        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        latency_ms = (time.perf_counter() - start) * 1000

        idx = int(np.argmax(probs))
        return {
            "fault_label": self.label_names[idx],
            "confidence": float(probs[idx]),
            "all_probabilities": {
                name: float(p) for name, p in zip(self.label_names, probs)
            },
            "latency_ms": latency_ms,
        }

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on a batch; returns (predicted_labels, mean_latency_ms)."""
        preds = []
        latencies = []
        for sample in X:
            result = self.predict(sample)
            preds.append(self.label_names.index(result["fault_label"]))
            latencies.append(result["latency_ms"])
        return np.array(preds), float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_model(
    h5_path: str | Path,
    tflite_path: str | Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict:
    """Compare .h5 vs .tflite on size, accuracy and latency."""
    h5_path = Path(h5_path)
    tflite_path = Path(tflite_path)

    # .h5 accuracy
    keras_model = tf.keras.models.load_model(h5_path)
    X_in = X_test[..., np.newaxis] if X_test.ndim == 3 else X_test
    keras_preds = keras_model.predict(X_in, verbose=0).argmax(axis=1)
    keras_acc = float((keras_preds == y_test).mean())

    # .tflite accuracy + latency
    clf = TFLiteClassifier(tflite_path, label_names)
    X_flat = X_test if X_test.ndim == 3 else X_test[..., 0]
    tflite_preds, mean_latency_ms = clf.predict_batch(X_flat)
    tflite_acc = float((tflite_preds == y_test).mean())

    h5_size_kb = h5_path.stat().st_size / 1024
    tflite_size_kb = tflite_path.stat().st_size / 1024

    return {
        "model": h5_path.stem,
        "h5_size_kb": round(h5_size_kb, 1),
        "tflite_size_kb": round(tflite_size_kb, 1),
        "size_reduction_pct": round((1 - tflite_size_kb / h5_size_kb) * 100, 1),
        "h5_accuracy": round(keras_acc, 4),
        "tflite_accuracy": round(tflite_acc, 4),
        "accuracy_delta": round(tflite_acc - keras_acc, 4),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "under_100kb": tflite_size_kb < 100,
        "under_50ms": mean_latency_ms < 50,
    }
