import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/model_comparison.json") as file:
    results = json.load(file)

models = list(results.keys())
metrics = ["accuracy", "precision", "recall", "f1_score"]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

for i, model in enumerate(models):
    values = [results[model][metric] * 100 for metric in metrics]

    positions = x + (i - 0.5) * width

    bars = plt.bar(
        positions,
        values,
        width,
        label=model
    )

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.2f}%",
            ha="center",
            fontsize=8
        )

plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1 Score"])
plt.ylabel("Score (%)")
plt.title("Baseline CNN vs MobileNetV2")
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()

plt.savefig("results/model_comparison.png", dpi=300)
plt.show()

print("Model comparison chart saved!")