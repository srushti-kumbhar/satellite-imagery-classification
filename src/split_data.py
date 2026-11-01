import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from preprocessing import load_dataset

RANDOM_STATE = 42

print("Loading EuroSAT dataset...")

X, y, classes = load_dataset()

print("Splitting dataset...")

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

os.makedirs("data/processed", exist_ok=True)

np.savez_compressed(
    "data/processed/train.npz",
    X=X_train,
    y=y_train
)

np.savez_compressed(
    "data/processed/validation.npz",
    X=X_val,
    y=y_val
)

np.savez_compressed(
    "data/processed/test.npz",
    X=X_test,
    y=y_test
)

with open("data/processed/classes.json", "w") as file:
    json.dump(classes, file, indent=4)

print("\nDATASET SPLITTING COMPLETED")
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))
print("Total images:", len(X_train) + len(X_val) + len(X_test))
print("Class names saved successfully!")