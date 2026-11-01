import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Loading dataset...")

train_data = np.load("data/processed/train.npz")
val_data = np.load("data/processed/validation.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Build transfer learning model
inputs = tf.keras.Input(shape=(64, 64, 3))

x = augmentation(inputs)
x = tf.keras.layers.Rescaling(2.0, offset=-1)(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_mobilenetv2.keras",
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

print("\nStarting MobileNetV2 training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

np.savez(
    "results/mobilenet_history.npz",
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"],
    loss=history.history["loss"],
    val_loss=history.history["val_loss"]
)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/mobilenet_accuracy.png")
plt.close()

print("\nMobileNetV2 training completed!")
print("Model saved: models/best_mobilenetv2.keras")