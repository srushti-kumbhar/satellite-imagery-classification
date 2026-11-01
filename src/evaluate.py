import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load test data
test_data = np.load("data/processed/test.npz")
X_test, y_test = test_data["X"], test_data["y"]

with open("data/processed/classes.json") as file:
    classes = json.load(file)

# Load trained CNN
model = tf.keras.models.load_model("models/best_cnn.keras")

print("Generating predictions...")
predictions = model.predict(X_test, batch_size=32, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nBASELINE CNN EVALUATION")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=classes,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.title("Baseline CNN Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/cnn_confusion_matrix.png")
plt.close()

# Save actual metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open("results/cnn_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nEvaluation completed!")
print("Confusion matrix saved.")
print("Metrics saved in results/cnn_metrics.json")