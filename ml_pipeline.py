import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_ROOT = "data"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR  = os.path.join(DATA_ROOT, "masks")

CLASS_NAMES = ["normal", "benign", "malignant"]

def get_label_from_mask(mask_path):
    mask = cv2.imread(mask_path)
    if mask is None:
        return 0
    b, g, r = cv2.split(mask)
    if np.mean(r) > np.mean(g) and np.mean(r) > np.mean(b):
        return 2
    elif np.mean(g) > np.mean(r) and np.mean(g) > np.mean(b):
        return 1
    else:
        return 0

def extract_features(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path)
    if img is None or mask is None:
        return None

    h, w = img.shape

    # lesion mask: any non-black pixel in mask
    lesion = (mask[:,:,1] > 10) | (mask[:,:,2] > 10)

    lesion_area = lesion.sum()
    lesion_ratio = lesion_area / (h * w)

    if lesion_area > 0:
        lesion_intensity = img[lesion].mean()
    else:
        lesion_intensity = 0.0

    whole_mean = img.mean()
    whole_std = img.std()

    return [lesion_ratio, lesion_intensity, whole_mean, whole_std]

if __name__ == "__main__":
    rows = []

    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith(".png"):
            continue
        img_path = os.path.join(IMAGE_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname)
        feats = extract_features(img_path, mask_path)
        if feats is None:
            continue
        label = get_label_from_mask(mask_path)
        rows.append(feats + [label])

    cols = ["lesion_ratio", "lesion_intensity", "img_mean", "img_std", "label"]
    df = pd.DataFrame(rows, columns=cols)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Classical ML (RandomForest) Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
