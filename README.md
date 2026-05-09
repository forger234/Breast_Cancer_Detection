# Breast Cancer Detection using Deep Learning & Machine Learning

## Overview

This project is an AI-powered medical diagnostic platform for **Breast Cancer Detection** using multiple imaging modalities including:

* Ultrasound Imaging
* Histopathology Imaging
* PET/CT Scan Analysis

The system combines **Deep Learning**, **Machine Learning**, and **Explainable AI (Grad-CAM)** to assist in early-stage breast cancer diagnosis and clinical interpretation.

The application provides:

* Real-time prediction
* Probability analysis
* Grad-CAM heatmaps
* PET/CT fusion visualization
* AI-generated interpretation
* Patient report generation (PDF)
* Doctor authentication system

---

# Features

## Ultrasound Classification

* Detects:

  * Normal
  * Benign
  * Malignant
* Uses ResNet18 Deep Learning architecture
* Grad-CAM explainability visualization

## Histopathology Analysis

* Histopathology image classification
* AI-assisted pathology interpretation

## PET/CT Analysis

* PET + CT image fusion
* Metabolic activity analysis
* Cancer stage estimation
* Clinical summary generation

## Patient Data Management

* Doctor login authentication
* Patient history storage
* PDF medical report generation

## Interactive Dashboard

* Built using Streamlit
* Modern medical UI
* Interactive probability charts
* Dynamic scan visualizations

---

# Technologies Used

| Category          | Technologies           |
| ----------------- | ---------------------- |
| Frontend          | Streamlit              |
| Deep Learning     | PyTorch, Torchvision   |
| Machine Learning  | Scikit-learn           |
| Image Processing  | OpenCV, PIL            |
| Medical Imaging   | Pydicom, Nibabel       |
| Visualization     | Plotly                 |
| AI Interpretation | Google Gemini API      |
| Report Generation | ReportLab              |
| Authentication    | JSON-based doctor auth |

---

# Project Structure

```bash
Breast-Cancer-Detection/
│
├── app_streamlit.py
├── requirements.txt
├── installed_requirements.txt
├── bact.jpg
│
├── models/
│   ├── breast_cancer_resnet18.pth
│   └── petct_model.pkl
│
├── auth/
│   ├── doctor_auth.py
│   └── doctor_users.json
│
├── histopathology/
│   └── histopathology_module.py
│
├── patient_data/
│   └── patient_data_module.py
│
├── petct/
│   └── petct_inference.py
│
├── dataset/
│
├── reports/
│
└── README.md
```

---

# Dataset Links

## 1. Breast Ultrasound Dataset

Dataset Source:

[Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?utm_source=chatgpt.com)

Classes:

* Normal
* Benign
* Malignant

---

## 2. Histopathology Dataset

Dataset Source:

[Breast Histopathology Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images?utm_source=chatgpt.com)

Contains:

* IDC Positive
* IDC Negative

---

## 3. PET/CT Dataset

Dataset Source:

[TCIA Breast Cancer PET/CT Collection](https://www.cancerimagingarchive.net/?utm_source=chatgpt.com)

Alternative Dataset:

[Medical PET/CT Imaging Dataset](https://www.kaggle.com/?utm_source=chatgpt.com)

---

# Model Architectures

## Ultrasound Model

* ResNet18
* Transfer Learning
* Grad-CAM Explainability

## Histopathology Model

* CNN-based classification model

## PET/CT Model

* Random Forest / ML Pipeline
* SUV-based metabolic analysis

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Requirements

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app_streamlit.py
```

Application will run at:

```bash
http://localhost:8501
```

---

# Screenshots

## Ultrasound Prediction
<img width="1912" height="960" alt="image" src="https://github.com/user-attachments/assets/e9e82a8b-569a-4cf7-8948-f6c506a1e0c5" />


## PET/CT Visualization

* CT Slice
* PET Metabolic Slice
* PET + CT Fusion

## Patient Report

* PDF report generation
* Clinical interpretation
* Doctor notes

---

# Explainable AI

This project integrates **Grad-CAM** to provide:

* Visual heatmaps
* Model attention regions
* Improved medical interpretability

---

# Security Features
<img width="1565" height="835" alt="image" src="https://github.com/user-attachments/assets/29e8d727-cbc4-4d25-b8e7-4971c8dea003" />



# Future Improvements

* Cloud database integration
* Multi-user hospital support
* Real-time DICOM PACS integration
* AI chatbot assistant
* Federated learning support
* Mobile application deployment

---

# Research Applications

This project can be used for:

* Medical AI research
* Breast cancer screening assistance
* Explainable healthcare AI
* Clinical decision support systems

---

# Disclaimer

This project is developed for:

* Educational purposes
* Research purposes
* AI-assisted diagnosis support

It should NOT replace professional medical diagnosis.

---

# Author

## Developed By

Your Name

## Project Type

Final Year Project / Research Project

---

# License

This project is licensed under the MIT License.

---

# Citation

If you use this project in research, please cite:

```text
AI-Powered Breast Cancer Detection using Deep Learning,
Machine Learning, and Multi-Modal Medical Imaging
```
