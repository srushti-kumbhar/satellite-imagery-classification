import os
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "data/EuroSAT_RGB"

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
    and not folder.startswith(".")
])

print("EUROSAT DATASET EXPLORATION")
print("=" * 40)
print("Total Classes:", len(classes))

total_images = 0
class_counts = {}

for category in classes:
    folder_path = os.path.join(DATASET_PATH, category)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts[category] = len(images)
    total_images += len(images)

    print(f"{category}: {len(images)} images")

print("=" * 40)
print("Total Images:", total_images)

# Class distribution chart
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("EuroSAT Dataset Class Distribution")
plt.xlabel("Terrain Categories")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/class_distribution.png")
plt.show()

print("Class distribution chart saved!")