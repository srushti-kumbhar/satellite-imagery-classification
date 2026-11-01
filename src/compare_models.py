import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = np.load("data/processed/test.npz")
X_test, y_test = test_data["X"], test_data["y"]

models = {
    "Baseline CNN": "models/best_cnn.keras",
    "MobileNetV2": "models/best_mobilenetv2.keras"
}

results = {}

for name, path in models.items():
    print(f"\nEvaluating {name}...")

    model = tf.keras.models.load_model(path)
    predictions = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    results[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    }

print("\nMODEL COMPARISON")
print("=" * 60)

for name, metrics in results.items():
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

with open("results/model_comparison.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nComparison saved to results/model_comparison.json")