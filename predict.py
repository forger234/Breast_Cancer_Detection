import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

IMAGE_SIZE = 256
CLASS_NAMES = ["normal", "benign", "malignant"]
MODEL_PATH = "breast_cancer_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]  # shape [3]
        top_idx = torch.argmax(probs).item()
        top_class = CLASS_NAMES[top_idx]
        top_prob = probs[top_idx].item()
    return top_class, top_prob, probs.cpu().numpy()

if __name__ == "__main__":
    # change this
    TEST_IMAGE_PATH = r"data\images\ORPE_026.png"

    model = load_model()
    cls, p, all_probs = predict_image(model, TEST_IMAGE_PATH)
    print(f"Prediction for {TEST_IMAGE_PATH}: {cls} ({p*100:.2f}% confidence)")
    print("All class probabilities:")
    for name, prob in zip(CLASS_NAMES, all_probs):
        print(f"  {name}: {prob*100:.2f}%")
