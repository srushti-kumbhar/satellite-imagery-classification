import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Loading training data...")

train_data = np.load("data/processed/train.npz")
val_data = np.load("data/processed/validation.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]

NUM_CLASSES = 10

# Baseline CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Save best model based on validation accuracy
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

print("\nStarting CNN training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# Save training history
np.savez(
    "results/cnn_history.npz",
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"],
    loss=history.history["loss"],
    val_loss=history.history["val_loss"]
)

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/cnn_accuracy.png")
plt.close()

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/cnn_loss.png")
plt.close()

print("\nCNN TRAINING COMPLETED!")
print("Best model saved in models/best_cnn.keras")
print("Training graphs saved in results/")