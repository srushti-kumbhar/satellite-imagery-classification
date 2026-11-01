import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = (64, 64)

with open("data/processed/classes.json") as file:
    classes = json.load(file)

model = tf.keras.models.load_model("models/best_mobilenetv2.keras")

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    print("\nSATELLITE IMAGE CLASSIFICATION")
    print("--------------------------------")
    print("Predicted category:", classes[predicted_index])
    print(f"Confidence: {confidence:.2f}%")

    print("\nTop 3 predictions:")
    top_indices = np.argsort(predictions)[-3:][::-1]

    for index in top_indices:
        print(f"{classes[index]}: {predictions[index] * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path_to_image")
    else:
        predict_image(sys.argv[1])