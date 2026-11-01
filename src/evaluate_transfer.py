import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load test dataset
test_data = np.load("data/processed/test.npz")
X_test, y_test = test_data["X"], test_data["y"]

with open("data/processed/classes.json") as file:
    classes = json.load(file)

# Load MobileNetV2
model = tf.keras.models.load_model("models/best_mobilenetv2.keras")

print("Generating predictions...")
predictions = model.predict(X_test, batch_size=32, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
}

print("\nMOBILENETV2 EVALUATION")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=classes,
    zero_division=0
))

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=classes,
    yticklabels=classes
)
plt.title("MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/mobilenet_confusion_matrix.png")
plt.close()

with open("results/mobilenet_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMobileNetV2 evaluation completed!")
print("Results saved in results/")