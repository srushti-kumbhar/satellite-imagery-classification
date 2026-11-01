import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "data/EuroSAT_RGB"

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
    and not folder.startswith(".")
])

random.seed(42)

plt.figure(figsize=(15, 10))

for i, category in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, category)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_name = random.choice(images)
    image_path = os.path.join(folder_path, image_name)

    image = Image.open(image_path).convert("RGB")

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(category)
    plt.axis("off")

plt.suptitle("EuroSAT Satellite Image Samples", fontsize=16)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/sample_images.png")
plt.show()

print("Sample image visualization completed!")