import os
import matplotlib.pyplot as plt
from PIL import Image

folder = "data/EuroSAT_RGB/AnnualCrop"
images = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:5]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for ax, filename in zip(axes, images):
    image = Image.open(os.path.join(folder, filename)).convert("RGB")
    ax.imshow(image, interpolation="nearest")
    ax.set_title(filename[:12])
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/original_resolution_preview.png", dpi=300)
plt.show()

print("Original-resolution images displayed.")