import os
import numpy as np
from PIL import Image

DATASET_PATH = "data/EuroSAT_RGB"
IMAGE_SIZE = (64, 64)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image, dtype=np.float32) / 255.0
    return image

def load_dataset():
    images = []
    labels = []

    classes = sorted([
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
        and not folder.startswith(".")
    ])

    class_to_index = {
        category: index
        for index, category in enumerate(classes)
    }

    for category in classes:
        folder_path = os.path.join(DATASET_PATH, category)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)

                try:
                    image = preprocess_image(image_path)
                    images.append(image)
                    labels.append(class_to_index[category])
                except Exception as error:
                    print("Skipping:", image_path, error)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y, classes


if __name__ == "__main__":
    X, y, classes = load_dataset()

    print("PREPROCESSING COMPLETED")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Number of classes:", len(classes))
    print("Categories:", classes)
    print("Pixel range:", X.min(), "to", X.max())