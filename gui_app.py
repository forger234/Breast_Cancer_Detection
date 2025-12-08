import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
import torch.nn.functional as F

IMAGE_SIZE = 256
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
MODEL_PATH = "breast_cancer_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
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

model = load_model()

def predict_image_with_prob(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    t = transform(pil)
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(t)
        probs = F.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        top_class = CLASS_NAMES[top_idx]
        top_prob = probs[top_idx].item()
    return top_class, top_prob

def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Ultrasound Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    cls, prob = predict_image_with_prob(file_path)
    result_label.config(text=f"Prediction: {cls} ({prob*100:.2f}% confidence)", fg="blue")

root = Tk()
root.title("Breast Cancer Detection System")
root.geometry("400x500")

title_label = Label(root, text="Breast Cancer Detection", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

upload_btn = Button(root, text="Upload Ultrasound Image", command=upload_image)
upload_btn.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="Prediction: ---", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
