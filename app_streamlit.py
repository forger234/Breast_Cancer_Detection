from petct.petct_inference import predict_petct
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import streamlit as st
import pydicom
import nibabel as nib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
genai.configure(api_key="")
from streamlit_lottie import st_lottie
import json
import time
import base64
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from datetime import datetime

from auth.doctor_auth import doctor_login

IMAGE_SIZE = 256
CLASS_NAMES = ["normal", "benign", "malignant"]
MODEL_PATH = "breast_cancer_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================= PAGE CONFIG & CUSTOM THEME =================
st.set_page_config(
    page_title="MammoSafe - Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    doctor_login()
    st.stop()

# ================= PATIENT HISTORY STORAGE =================

HISTORY_DIR = "patient_records"
HISTORY_FILE = os.path.join(HISTORY_DIR, "patient_history.csv")

def save_patient_history(
    patient_name,
    patient_id,
    prediction,
    confidence,
    stage,
    risk_score
):

    os.makedirs(HISTORY_DIR, exist_ok=True)

    new_data = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Date": datetime.now().strftime("%d-%m-%Y"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": prediction,
        "Confidence (%)": round(confidence, 2),
        "Risk Score": round(risk_score, 2),
        "Stage": stage
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)

        # 🚫 Avoid duplicate Patient ID
        if patient_id in df["Patient ID"].values:
            
            return  # do not save again

        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(HISTORY_FILE, index=False)


import uuid

# (Page config moved to top)
st.markdown("""
<style>

/* ===== Radio group title ===== */
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] p,
div[data-testid="stRadio"] > label {
    color: #ffffff !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}

/* ===== Radio option text ===== */
div[data-testid="stRadio"] [role="radiogroup"] label p,
div[data-testid="stRadio"] [role="radiogroup"] label {
    color: #ffffff !important;        /* FORCE WHITE */
    font-size: 1.25rem !important;    /* Bigger text */
    font-weight: 700 !important;
}

/* ===== Selected option ===== */
div[data-testid="stRadio"] [role="radiogroup"] label input:checked ~ div p,
div[data-testid="stRadio"] [role="radiogroup"] label input:checked ~ div {
    color: #22c55e !important;        /* Green selected */
    font-weight: 900 !important;
}

/* ===== Hover ===== */
div[data-testid="stRadio"] [role="radiogroup"] label:hover p,
div[data-testid="stRadio"] [role="radiogroup"] label:hover {
    color: #fde047 !important;        /* Yellow hover */
}

/* ===== Radio circles ===== */
div[data-testid="stRadio"] svg {
    fill: #ffffff !important;
}

div[data-testid="stRadio"] [role="radiogroup"] label input:checked ~ div svg {
    fill: #22c55e !important;
}

/* ===== Metric Sizes ===== */
[data-testid="stMetricValue"] > div {
    font-size: 1.5rem !important; /* Smaller Value */
}
[data-testid="stMetricLabel"] > div > div > p {
    font-size: 0.9rem !important; /* Smaller Label */
}

/* ===== Alert Text Color (st.info, st.error, etc) ===== */
div[data-testid="stAlert"] p, 
div[data-testid="stAlert"] div {
    color: #ffffff !important;
}

/* ===== Patient History Box (Expanders & DataFrames) ===== */
div[data-testid="stExpander"] details summary p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
div[data-testid="stExpanderDetails"] {
    color: #ffffff !important;
}
/* Force DataFrame and Table cells to white */
div[data-testid="stDataFrame"] div, 
div[data-testid="stDataFrame"] th, 
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] span {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown(f"👨‍⚕️ Logged in as: **{st.session_state.doctor_name}**")

    st.markdown("---")

    st.markdown("### 🏷️ Classification Labels")
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="status-badge normal">Normal</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="status-badge benign">Benign</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="status-badge malignant">Malignant</div>', unsafe_allow_html=True)

    st.markdown("---")


   

    # ✅ Logout placed directly under dataset info
    if st.button(" Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.doctor_name = None
        st.rerun()
# Function to encode local image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to add background image
def add_bg_image():
    # Try to load local bact.jpg, fallback to online image if not found
    try:
        # Check if local file exists
        if os.path.exists("./bact.jpg"):
            bg_image = f"data:image/jpg;base64,{get_base64_of_bin_file('bact.jpg')}"
        else:
            # Fallback to online image
            bg_image = "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
    except:
        bg_image = "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
    
    st.markdown(
        f"""
        <style>
            /* Banner section with background image */
            .banner-section {{
                background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                            url('{bg_image}');
                background-size: cover;
                background-position: center;
                padding: 80px 40px;
                border-radius: 20px;
                margin-bottom: 40px;
                text-align: center;
                animation: fadeIn 1s ease-out;
            }}
            
            .banner-title {{
                font-size: 3.5rem;
                font-weight: 900;
                color: white;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                letter-spacing: 1px;
            }}
            
            .banner-subtitle {{
                font-size: 1.5rem;
                color: #e2e8f0;
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
            }}
            
            /* Modern gradient background with subtle animation */
            .stApp {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
                color: #f1f5f9;
            }}
            
            @keyframes gradientShift {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            
            /* Smooth fade-in animation */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            /* Title styling with animation */
            .title-text {{
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(90deg, #60a5fa, #34d399, #fbbf24);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                animation: fadeIn 1s ease-out;
                text-align: center;
                margin-bottom: 0.5rem;
            }}
            
            .subtitle-text {{
                font-size: 1.1rem;
                color: #cbd5e1;
                text-align: center;
                animation: fadeIn 1.2s ease-out;
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
            }}
            
            /* File uploader text color modifications - GREEN for all text */
            [data-testid="stFileUploader"] label {{
                color: #22c55e !important;
                font-size: 1.2rem !important;
                font-weight: 600 !important;
            }}
            
            /* File uploader instruction text */
            [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
                color: #86efac !important;
                font-size: 0.95rem !important;
            }}
            
            /* File name display styling - GREEN */
            .uploadedFileName {{
                color: #22c55e !important;
                font-weight: 600 !important;
                background: rgba(34, 197, 94, 0.1) !important;
                padding: 8px 16px !important;
                border-radius: 10px !important;
                border: 1px solid rgba(34, 197, 94, 0.3) !important;
                margin-top: 10px !important;
            }}
            
            /* Browse files button styling - GREEN */
            [data-testid="stFileUploader"] button {{
                background: linear-gradient(45deg, #22c55e, #16a34a) !important;
                color: white !important;
                border: none !important;
                padding: 10px 24px !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }}
            
            [data-testid="stFileUploader"] button:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4) !important;
                background: linear-gradient(45deg, #16a34a, #22c55e) !important;
            }}
            
            /* File list items - GREEN */
            [data-testid="stFileUploader"] [data-testid="stFileDropzoneInstructions"] div {{
                color: #86efac !important;
            }}
            
            /* Glassmorphism card with hover effects */
            .glass-card {{
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 25px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
                animation: fadeIn 0.8s ease-out;
            }}
            .equal-card {{
                    min-height: 280px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }}
            .glass-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
                border-color: rgba(96, 165, 250, 0.3);
            }}
            
            /* Metric card with gradient border */
            .metric-card {{
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
                border-radius: 18px;
                padding: 25px;
                border: 1px solid;
                border-image: linear-gradient(45deg, #60a5fa, #34d399) 1;
                position: relative;
                overflow: hidden;
                animation: fadeIn 1s ease-out;
            }}
            
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                transition: 0.5s;
            }}
            
            .metric-card:hover::before {{
                left: 100%;
            }}
            
            /* Enhanced probability bars */
            .prob-bar-container {{
                margin: 15px 0;
            }}
            
            .prob-bar {{
                height: 14px;
                border-radius: 10px;
                background: rgba(30, 41, 59, 0.8);
                overflow: hidden;
                position: relative;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            
            .prob-fill {{
                height: 100%;
                border-radius: 10px;
                background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
                transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }}
            
            .prob-fill::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, 
                    transparent 0%, 
                    rgba(255, 255, 255, 0.2) 50%, 
                    transparent 100%);
                animation: shimmer 2s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            /* File uploader styling */
            [data-testid="stFileUploader"] > div:first-child {{
                border-radius: 16px;
                border: 2px dashed rgba(34, 197, 94, 0.4);
                background-color: rgba(15, 23, 42, 0.5);
                transition: all 0.3s ease;
                padding: 40px 20px;
            }}
            
            [data-testid="stFileUploader"] > div:first-child:hover {{
                border-color: #22c55e;
                background-color: rgba(15, 23, 42, 0.7);
            }}
            
            /* Button styling */
            .stButton > button[kind="primary"] {{
                background: linear-gradient(45deg, #60a5fa, #34d399);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
            }}
            
            .stButton > button[kind="primary"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(96, 165, 250, 0.4);
            }}

            /* Professional secondary button styling */
            .stButton > button[kind="secondary"],
            .stDownloadButton > button {{
                background: rgba(30, 41, 59, 0.6);
                color: #ffffff !important;
                border: 1px solid rgba(148, 163, 184, 0.4);
                padding: 12px 24px;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
            }}

            .stButton > button[kind="secondary"]:hover,
            .stDownloadButton > button:hover {{
                background: rgba(45, 55, 72, 0.8);
                color: #ffffff !important;
                border-color: #60a5fa;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2);
            }}
            
            /* Ensure download button SVG icons are white */
            .stDownloadButton > button svg {{
                fill: #ffffff !important;
                color: #ffffff !important;
            }}

            /* Metrics text colors */
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricLabel"] > div {{
                color: #94a3b8 !important;   /* Slate 400 */
                font-weight: 600 !important;
            }}
            
            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] div {{
                color: #ffffff !important;   /* White */
                font-weight: 800 !important;
            }}
            
            /* Expander text color override */
            [data-testid="stExpander"] summary span,
            [data-testid="stExpander"] summary p {{
                color: #0f172a !important; /* Make the white text dark so it's visible on default backgrounds */
                font-weight: 700;
            }}
            
            [data-testid="stExpander"] svg {{
                color: #0f172a !important;
                fill: #0f172a !important;
            }}

            /* Status badges */
            .status-badge {{
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9rem;
                margin: 5px;
                animation: pulse 2s infinite;
            }}
            
            .normal {{ background: linear-gradient(45deg, #10b981, #34d399); }}
            .benign {{ background: linear-gradient(45deg, #f59e0b, #fbbf24); }}
            .malignant {{ background: linear-gradient(45deg, #ef4444, #f87171); }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.8; }}
            }}
            
            /* Image containers */
            .image-container {{
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }}
            
            .image-container:hover {{
                transform: scale(1.02);
            }}
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: rgba(30, 41, 59, 0.5);
                border-radius: 12px 12px 0 0;
                padding: 12px 24px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                color: #cbd5e1;
                transition: all 0.3s ease;
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background-color: rgba(30, 41, 59, 0.8);
                color: white;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(45deg, #60a5fa, #34d399) !important;
                color: white !important;
                border-color: transparent !important;
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 10px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: rgba(15, 23, 42, 0.5);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(45deg, #60a5fa, #34d399);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(45deg, #34d399, #60a5fa);
            }}
            
            /* Section headers */
            .section-header {{
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(90deg, #60a5fa, #34d399);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid rgba(96, 165, 250, 0.3);
            }}
            
        </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_image()

# ================= CONFIG =================

def load_dicom_series(files):
    slices = []

    for f in files:
        ds = pydicom.dcmread(f)
        slices.append(ds)

    # Sort by slice position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    return volume, slices[0]


def process_ct(img, ds):
    hu = img * ds.RescaleSlope + ds.RescaleIntercept
    hu = np.clip(hu, -150, 250)
    hu = (hu - hu.min()) / (hu.max() - hu.min())
    return (hu * 255).astype(np.uint8)

def process_pet(img):
    img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)

def fuse_pet_ct(ct_img, pet_img):
    ct_rgb = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2RGB)
    pet_heat = cv2.applyColorMap(pet_img, cv2.COLORMAP_JET)
    return cv2.addWeighted(ct_rgb, 0.6, pet_heat, 0.4, 0)

DATA_ROOT = "data"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")



img_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

def get_ai_interpretation_text(pred_name, confidence):
    if pred_name == "normal":
        ultrasound = (
            "The ultrasound image shows homogeneous tissue structure, "
            "smooth boundaries, and no visible lesion mass."
        )
        explanation = (
            "The AI model analyzed spatial texture patterns and deep feature activations. "
            "Grad-CAM visualization confirms that the prediction is driven by lesion-specific "
            "regions rather than background tissue."
        )
        followup = "No immediate follow-up required."

    elif pred_name == "benign":
        ultrasound = (
            "The ultrasound image shows a well-defined lesion with smooth margins "
            "and uniform internal texture."
        )
        explanation = (
            "The AI model focused on localized lesion regions with consistent internal patterns. "
            "Grad-CAM highlights benign structural features."
        )
        followup = (
            "Periodic clinical follow-up and ultrasound monitoring are recommended."
        )

    else:  # malignant
        ultrasound = (
            "The ultrasound image shows an irregular hypoechoic mass with heterogeneous texture "
            "and poorly defined margins."
        )
        explanation = (
            "The AI model detected abnormal spatial features and strong activations "
            "in suspicious lesion regions. Grad-CAM confirms malignant characteristics."
        )
        followup = (
            "Immediate clinical consultation is strongly recommended. "
            "Further diagnostic procedures such as biopsy or MRI are advised."
        )

    return ultrasound, explanation, followup

def draw_multiline_text(c, text, x, y, max_width, leading=14):
    """
    Draws wrapped multi-line text in ReportLab.
    """
    textobject = c.beginText(x, y)
    textobject.setLeading(leading)

    words = text.split(" ")
    line = ""

    for word in words:
        test_line = line + word + " "
        if c.stringWidth(test_line, "Helvetica", 11) <= max_width:
            line = test_line
        else:
            textobject.textLine(line)
            line = word + " "
    if line:
        textobject.textLine(line)

    c.drawText(textobject)
    return textobject.getY()

def generate_patient_report(
    patient_name,
    patient_id,
    patient_age,
    prediction,
    confidence,
    stage,
    probs,
    us_pred=None,
    us_conf=None,
    histo_pred=None,
    histo_conf=None,
    doctor_notes=""
):

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 50
    max_width = width - 2 * margin_x
    y = height - 50

    # ===== TITLE =====
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin_x, y, "Breast Cancer Diagnosis Report")
    y -= 40

    # ===== PATIENT INFO =====
    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Patient Name: {patient_name}")
    y -= 20
    c.drawString(margin_x, y, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    y -= 20
    c.drawString(margin_x, y, f"Patient ID: {patient_id}")
    y -= 18
    c.drawString(margin_x, y, f"Age: {patient_age} years")
    y -= 18
    c.line(margin_x, y, width - margin_x, y)
    y -= 30
    # ===== AI SUMMARY =====
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "AI Prediction Summary")
    y -= 20

    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Predicted Class: {prediction.upper()}")
    y -= 18
    c.drawString(margin_x, y, f"Confidence Score: {confidence:.2f}%")
    y -= 18
    # Make predicted stage standalone and bold to emphasize "stage 1", "stage 2"
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, f"Cancer Stage: {stage}")
    y -= 30

    # ===== CLASS PROBABILITIES =====
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Class Probabilities:")
    y -= 18

    c.setFont("Helvetica", 11)
    for cls, p in zip(["Normal", "Benign", "Malignant"], probs):
        c.drawString(margin_x + 20, y, f"{cls}: {p*100:.2f}%")
        y -= 16

    y -= 20
    
    # ===== INDIVIDUAL MODALITIES =====
    if us_pred and histo_pred:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_x, y, "Individual Modality Predictions:")
        y -= 18
        
        c.setFont("Helvetica", 11)
        c.drawString(margin_x + 20, y, f"Ultrasound: {us_pred.upper()} ({us_conf:.2f}%)")
        y -= 16
        c.drawString(margin_x + 20, y, f"Histopathology: {histo_pred.upper()} ({histo_conf:.2f}%)")
        y -= 20

    c.line(margin_x, y, width - margin_x, y)
    y -= 30

    # ===== IMPRESSION =====
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Impression:")
    y -= 20
    
    impression_text = (
        f"Based on the combined analysis of Ultrasound and Histopathology modalities, "
        f"the patient presents with a finding consistent with <font name='Helvetica-Bold'>{prediction.upper()}</font> pathology, "
        f"indicative of <font name='Helvetica-Bold'>{stage}</font>."
    )
    
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = "Helvetica"
    style.fontSize = 11
    style.leading = 14
    
    p = Paragraph(impression_text, style)
    w, h = p.wrap(max_width, height)
    p.drawOn(c, margin_x, y - h)
    y -= (h + 20)

    # ===== DOCTOR's NOTES / COMMENTS =====
    if doctor_notes and doctor_notes.strip():
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y, "Doctor's Notes & Recommendations")
        y -= 20
        
        c.setFont("Helvetica", 11)
        y = draw_multiline_text(c, doctor_notes.strip(), margin_x, y, max_width)
        y -= 20

    # ===== FOOTER =====
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        margin_x,
        40,
        "This report is generated by an MammoSafe."
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ================= LOAD LOTTIE ANIMATION =================

def load_lottie(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# ================= MODEL =================

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ================= GRAD-CAM CORE =================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        acts = self.activations[0]
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()

        return cam, output

def get_gradcam_overlay(pil_img, model):
    rgb = np.array(pil_img.convert("RGB"))
    t = img_transform(pil_img).unsqueeze(0).to(DEVICE)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam, output = gradcam.generate(t)
    
    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    
    h, w, _ = rgb.shape
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(cam_resized * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.42 * heatmap_rgb + 0.58 * rgb)

    return overlay, pred_name, probs

# ================= BANNER SECTION =================

st.markdown(
    """
    <div class="banner-section">
        <div class="banner-title">MammoSafe</div>
        <div class="banner-subtitle">
            Advanced AI-powered diagnostic platform integrating Ultrasound, Histopathology, and PET/CT analysis for comprehensive breast cancer detection.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================= SCAN TYPE SELECTION & UPLOAD =================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

col_title, col_refresh = st.columns([5,1])
with col_title:
    st.markdown("## 🔍 Select Scan Modality")
with col_refresh:
    if st.button("🔄 Refresh"):
        # Keep login info safe
        logged_in = st.session_state.get("logged_in", False)
        doctor_name = st.session_state.get("doctor_name", None)

        # Clear everything
        st.session_state.clear()

        # Restore login session
        st.session_state.logged_in = logged_in
        st.session_state.doctor_name = doctor_name
        st.session_state.uploader_key = str(uuid.uuid4())

        st.rerun()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# Sidebar Logo/Menu Code ...

st.markdown("""
<style>
    /* Traditional Tab Styling for Radio Buttons */
    div.row-widget.stRadio > div { 
        flex-direction: row; 
        align-items: stretch; 
        justify-content: flex-start;
        background: transparent;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        padding: 0; 
        margin-bottom: 20px;
        gap: 0;
    }
    div.row-widget.stRadio > div > label { 
        padding: 10px 20px; 
        background: transparent; 
        border-radius: 8px 8px 0 0; 
        cursor: pointer; 
        text-align: center;
        transition: all 0.2s ease;
        border: none;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: -2px; /* Pull down to overlap border */
    }
    div.row-widget.stRadio > div > label:hover {
        color: #e2e8f0;
        background: rgba(255,255,255,0.05);
    }
    div.row-widget.stRadio > div > label[data-checked="true"] { 
        background: transparent !important; 
        color: #3b82f6 !important;
        font-weight: 600;
        border-bottom: 2px solid #3b82f6;
        box-shadow: none;
    }
    
    /* Hide the default radio circle */
    div.row-widget.stRadio > div > label > div:first-child { display: none; }
</style>
""", unsafe_allow_html=True)

active_tab = st.radio(
    "Select Scan Modality",
    ["Ultrasound", "PET / CT", "Histopathology", "Patient Data"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded = None
uploaded_files = None
uploaded_us = None
uploaded_histo = None

if active_tab == "Ultrasound":
    st.markdown("### 📤 Upload Ultrasound Scan")
    uploaded = st.file_uploader(
        "Upload Ultrasound Image",
        type=["png", "jpg", "jpeg"],
        key=f"ultrasound_{st.session_state.uploader_key}"
    )
    if uploaded:
        st.success(f"Uploaded: {uploaded.name}")

elif active_tab == "PET / CT":
    st.markdown("### 📤 Upload PET/CT Scan")
    uploaded_files = st.file_uploader(
        "Upload PET/CT DICOM Folder (select all files)",
        type=["dcm"],
        accept_multiple_files=True,
        key=f"petct_{st.session_state.uploader_key}"
    )
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} DICOM files")

elif active_tab == "Histopathology":
    st.markdown("### 🔬 Histopathology Scan")
    uploaded = st.file_uploader(
        "Upload Histopathology Image",
        type=["png", "jpg", "jpeg"],
        key=f"histo_{st.session_state.uploader_key}"
    )
    if uploaded:
        st.success(f"Uploaded: {uploaded.name}")

elif active_tab == "Patient Data":
    st.markdown("### 📊 Patient Data Uploading Purpose")
    
    uploaded_folder_files = st.file_uploader(
        "Upload Patient Folder",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"patient_folder_{st.session_state.uploader_key}"
    )
    
    if uploaded_folder_files and len(uploaded_folder_files) >= 2:
        st.success(f"Selected {len(uploaded_folder_files)} files.")
        
        # Sort so they're in a consistent order
        uploaded_folder_files = sorted(uploaded_folder_files, key=lambda x: x.name)
        
        # Determine default index based on filename
        us_index = 0
        histo_index = 1 if len(uploaded_folder_files) > 1 else 0
        
        for i, f in enumerate(uploaded_folder_files):
            if "us" in f.name.lower():
                us_index = i
            elif "histo" in f.name.lower():
                histo_index = i
                
        col1, col2 = st.columns(2)
        with col1:
            uploaded_us = st.selectbox(
                "", 
                uploaded_folder_files, 
                format_func=lambda x: x.name, 
                index=us_index
            )
        with col2:
            uploaded_histo = st.selectbox(
                "", 
                uploaded_folder_files, 
                format_func=lambda x: x.name, 
                index=histo_index
            )
    elif uploaded_folder_files:
        st.warning("Please select at least 2 images.")

scan_type = active_tab

# ================= PROCESSING =================

if (scan_type in ["Ultrasound", "Histopathology"] and uploaded is not None) or \
   (scan_type == "PET / CT" and uploaded_files) or \
   (scan_type == "Patient Data" and uploaded_folder_files and len(uploaded_folder_files) >= 2):

    with st.spinner("  Processing scan..."):

        # ---------- ULTRASOUND ----------
        if scan_type == "Ultrasound":
            pil_img = Image.open(uploaded).convert("RGB")
            # Validation: Ultrasound should be mostly grayscale
            img_arr = np.array(pil_img).astype(float)
            r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
            color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))
            if color_diff > 50: # Significant color variance (>50) means it's likely Histopathology
                st.error("⚠️ Invalid image format. Please upload an Ultrasound image.")
                st.stop()
                
            model = load_model()
            overlay_np, pred_name, probs = get_gradcam_overlay(pil_img, model)

        # ---------- HISTOPATHOLOGY ----------
        elif scan_type == "Histopathology":
            from histopathology.histopathology_module import predict_histopathology
            pil_img = Image.open(uploaded).convert("RGB")
            # Validation: Histopathology should be colored (H&E stain)
            img_arr = np.array(pil_img).astype(float)
            r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
            color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))
            if color_diff < 50: # Low color variance (<50) means it's likely Ultrasound/grayscale
                st.error("⚠️ Invalid image format.Please upload a Histopathology image.")
                st.stop()
                
            pred_name, confidence, probs_list = predict_histopathology(pil_img)
            # Reorder probs to match CLASS_NAMES ["normal", "benign", "malignant"]
            probs = np.zeros(3)
            probs[0] = probs_list[2] # Normal
            probs[1] = probs_list[0] # Benign
            probs[2] = probs_list[1] # Malignant

        # ---------- PET / CT ----------
        elif scan_type == "PET / CT":
            try:
                result = predict_petct(uploaded_files)

                st.markdown('<div class="section-header">🧠 PET / CT Clinical Assessment</div>', unsafe_allow_html=True)

                    # Top Row: Stage & Summary vs Metrics
                top_col1, top_col2 = st.columns([1.5, 1])
                
                with top_col1:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="height: 100%;">
                            <h3 style="color: #60a5fa; margin-top:0;">🩺 Estimated Stage</h3>
                            <div class="status-badge malignant" style="font-size: 1.5rem; padding: 10px 20px; display:inline-block; margin-bottom: 20px;">
                                {result['stage']}
                            </div>
                            <h4 style="color: #cbd5e1;">📋 Clinical Summary</h4>
                            <p style="color: #e2e8f0; line-height: 1.6;">{result['clinical_summary']}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                with top_col2:
                    st.markdown(
                        """
                        <div class="glass-card" style="height: 100%;">
                            <h3 style="color: #34d399; margin-top:0;">📊 Metabolic Metrics</h3>
                        """, 
                        unsafe_allow_html=True
                    )
                    metrics = result["metrics"]
                    for k, v in metrics.items():
                        st.metric(k, f"{v:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)

                # ================= GEMINI AI ORGAN IDENTIFICATION =================
                st.markdown('<div class="section-header">🤖 AI Organ Identification</div>', unsafe_allow_html=True)
                with st.spinner("Analyzing anatomical features..."):
                    try:
                        # Convert the CT slice to a PIL Image for the VLM
                        ct_img_array = result["images"]["CT Slice (Anatomical)"]
                        # If it's a grayscale image, convert to RGB for PIL
                        if len(ct_img_array.shape) == 2:
                            ct_img_array = cv2.cvtColor(ct_img_array, cv2.COLOR_GRAY2RGB)
                        pil_ct_img = Image.fromarray(ct_img_array)

                        # Initialize the model and generate content
                        vlm_model = genai.GenerativeModel('gemini-2.5-flash')
                        prompt = (
                            "You are an expert radiologist. Analyze this medical CT slice from a PET/CT scan. "
                            "Identify the anatomical region (e.g., chest, abdomen, pelvis), describe the organ details visible, "
                            "and point out any general characteristics. Keep your response professional, concise, and structured."
                        )
                        response = vlm_model.generate_content([prompt, pil_ct_img])
                        
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border-left: 4px solid #a855f7;">
                                <h4 style="color: #a855f7; margin-top:0;">AI Analysis</h4>
                                <p style="color: #e2e8f0; line-height: 1.6; font-size: 1.1rem;">
                                    {response.text}
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"AI Analysis Failed: {str(e)}")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Visual Interpretation</div>', unsafe_allow_html=True)
                
                # Images Row
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(
                        result["images"]["CT Slice (Anatomical)"],
                        caption="CT Slice (Anatomical)",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(
                        result["images"]["PET Slice (Metabolic)"],
                        caption="PET Slice (Metabolic)",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with c3:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(
                        result["images"]["PET + CT Fusion"],
                        caption="PET + CT Fusion",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Dynamic Active Region Marker
                active_pct = metrics.get("Active region (%)", 0)
                
                # Map actual percentage to visual position (simplified mapping):
                # 0-5% is Low (0-33%)
                # 5-15% is Moderate (33-66%)
                # >15% is High (66-100%)
                if active_pct <= 5:
                    left_pos = (active_pct / 5.0) * 33.3
                elif active_pct <= 15:
                    left_pos = 33.3 + ((active_pct - 5) / 10.0) * 33.3
                else:
                    left_pos = 66.6 + min(((active_pct - 15) / 20.0) * 33.3, 33.3) # cap at 100%
                
                left_pos = max(2, min(left_pos, 98)) # keep marker inside bounds
                
                marker_html = f"""<div style="position: absolute; left: {left_pos}%; top: -25px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; z-index: 10;">
<div style="background: rgba(0,0,0,0.8); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; white-space: nowrap; margin-bottom: 2px; border: 1px solid rgba(255,255,255,0.3);">
{active_pct:.1f}% Area
</div>
<div style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 8px solid white;"></div>
</div>"""

                # Doctor interpretation explanation
                st.markdown(
                    "<div class='glass-card'>"
                    "<h4 style='color: #fbbf24; margin-top:0;'>👨‍⚕️ Interpretation Guide</h4>"
                    "<div style='display: flex; gap: 20px; color: #cbd5e1; margin-bottom: 20px;'>"
                    "<div style='flex: 1;'><strong>• PET:</strong> Highlights metabolic activity (FDG uptake)</div>"
                    "<div style='flex: 1;'><strong>• CT:</strong> Shows physical anatomical structure</div>"
                    "<div style='flex: 1;'><strong>• Fusion:</strong> Correlates hotspots with anatomy</div>"
                    "<div style='flex: 1;'><strong>• Stage:</strong> Inferred from uptake intensity & spread</div>"
                    "</div>"
                    "<div style='margin-top: 25px; margin-bottom: 20px; padding: 20px; background: rgba(15, 23, 42, 0.4); border-radius: 12px; border: 1px solid rgba(52, 211, 153, 0.2);'>"
                    "<div style='margin-bottom: 30px; text-align: center;'>"
                    "<span style='color: #34d399; font-size: 1.3rem; font-weight: bold;'>🎨 PET & Fusion Disease Area Indicator</span>"
                    "</div>"
                    "<div style='position: relative; margin-bottom: 15px; margin-top: 25px;'>"
                    f"{marker_html}"
                    "<div style='height: 24px; border-radius: 12px; background: linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000); box-shadow: inset 0 2px 4px rgba(0,0,0,0.3), 0 2px 8px rgba(0,0,0,0.2);'></div>"
                    "</div>"
                    "<div style='display: flex; justify-content: space-between; font-size: 1rem; font-weight: 500; font-family: monospace;'>"
                    "<div style='text-align: left; width: 33%;'>"
                    "<span style='color: #93c5fd; font-size: 1.0rem;'>Low / Normal</span><br>"
                    "<span style='font-size: 0.85rem; color: #94a3b8;'>(Dark & Light Blue)</span>"
                    "</div>"
                    "<div style='text-align: center; width: 33%;'>"
                    "<span style='color: #fef08a; font-size: 1.0rem;'>Moderate</span><br>"
                    "<span style='font-size: 0.85rem; color: #94a3b8;'>(Green & Yellow)</span>"
                    "</div>"
                    "<div style='text-align: right; width: 33%;'>"
                    "<span style='color: #fca5a5; font-size: 1.0rem;'>High / Suspicious</span><br>"
                    "<span style='font-size: 0.85rem; color: #94a3b8;'>(Red & Dark Red)</span>"
                    "</div>"
                    "</div>"
                    "</div>"
                    "</div>", 
                    unsafe_allow_html=True
                )

                st.stop()

            except Exception as e:
                st.error(f"PET/CT processing error: {e}")
                st.stop()


        # ---------- PATIENT DATA UPLOADING PURPOSE ----------
        elif scan_type == "Patient Data":
            from patient_data.patient_data_module import predict_patient_data
            us_pil_img = Image.open(uploaded_us).convert("RGB")
            histo_pil_img = Image.open(uploaded_histo).convert("RGB")
            model = load_model()
            
            patient_results = predict_patient_data(
                us_pil_img, 
                histo_pil_img, 
                model, 
                get_gradcam_overlay, 
                CLASS_NAMES
            )
            
            # Extract combined results to be used in the common UI below
            pred_name = patient_results["combined"]["prediction"]
            confidence = patient_results["combined"]["confidence"]
            probs = patient_results["combined"]["probabilities"]
            
            us_pred_name = patient_results["ultrasound"]["prediction"]
            us_confidence = patient_results["ultrasound"]["confidence"]
            us_overlay_np = patient_results["ultrasound"]["overlay"]
            
            histo_pred_name = patient_results["histopathology"]["prediction"]
            histo_confidence = patient_results["histopathology"]["confidence"]

    # ========= PREDICTION RESULTS =========
    if scan_type in ["Ultrasound", "Histopathology", "Patient Data"]:
        st.markdown('<div class="section-header">📈 Analysis Results</div>', unsafe_allow_html=True)
        
        # Prediction card with enhanced visuals
        pred_index = CLASS_NAMES.index(pred_name)
        if scan_type == "Ultrasound":
            confidence = probs[pred_index] * 100.0
        # For Histopathology and Patient Data Uploading Purpose, confidence is already percentage
        
        # Risk score calculation
        if pred_name == "malignant":
            risk_score = confidence
        elif pred_name == "benign":
            risk_score = confidence * 0.6
        else:
            risk_score = confidence * 0.2
        
        col_left, col_mid, col_right = st.columns([1.5, 2, 1.5])
        
        with col_mid:
            st.markdown(
                f"""
                <div class="metric-card" style="padding: 40px; text-align: center;">
                    <div style="margin-bottom: 30px;">
                        <div style="font-size: 2.2rem; font-weight: 800; margin-bottom: 15px;">
                            Prediction
                        </div>
                        <div class="status-badge {pred_name}" style="font-size: 3.5rem; padding: 12px 30px; letter-spacing: 1px;">
                            {pred_name.upper()}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 2.0rem; font-weight: 900; color: #60a5fa; margin: 15px 0;">
                            {confidence:.1f}%
                        </div>
                        <div style="color: #cbd5e1; font-size: 1.2rem; font-weight: 500;">
                            Confidence Score
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Probability distribution with Plotly
        st.markdown("### 📊 Probability Distribution")
        
        # Create interactive chart
        fig = go.Figure(data=[
            go.Bar(
                x=[c.upper() for c in CLASS_NAMES],
                y=probs * 100,
                marker_color=['#10b981', '#f59e0b', '#ef4444'],
                text=[f'{p*100:.2f}%' for p in probs],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title='Classes'),
            yaxis=dict(title='Probability (%)', range=[0, 100]),
            height=400,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # ========= VISUAL EXPLANATIONS =========
        st.markdown('<div class="section-header">🔍 Visual Explanations</div>', unsafe_allow_html=True)
        
        if scan_type == "Patient Data":
            st.markdown("### Combined Modality Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Ultrasound: {us_pred_name.upper()} ({us_confidence:.1f}%)")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(us_pil_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Original Ultrasound Image")
                
            with col2:
                st.markdown(f"#### Histopathology: {histo_pred_name.upper()} ({histo_confidence:.1f}%)")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(histo_pil_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Original Histopathology Image")
                
        # Tabs for different views (conditionally hide heatmap if normal)
        elif pred_name == "normal" or scan_type == "Histopathology":
            # Only show original image
            #st.info("💡 Grad-CAM Heatmap visualization is not shown for Normal predictions as there are no specific lesion heat signatures to flag.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption(f"Original {scan_type} Image")
        else:
            tab1, tab3 = st.tabs(["🖼️ Original", "🔥 Grad-CAM Heatmap"])
            
            with tab1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption(f"Original {scan_type} Image")
            
            with tab3:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(overlay_np, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("Grad-CAM Heatmap - Areas of model attention")

        # ================= PATIENT REPORT =================
        if scan_type == "Patient Data":
            if "patient_id" not in st.session_state:
                st.session_state.patient_id = f"PID-{uuid.uuid4().hex[:8].upper()}"

            st.markdown('<div class="section-header">🧾 Patient Report</div>', unsafe_allow_html=True)
            st.markdown("### 👤 Patient Information")

            st.markdown(
                """
                <style>
                /* Text input fields and number inputs */
                div[data-baseweb="input"] > input,
                div[data-baseweb="base-input"] > input {
                    color: #000000 !important;
                    background-color: #ffffff !important;
                    border-radius: 4px !important;
                }
                
                /* Placeholder text color */
                input::placeholder {
                    color: #94a3b8 !important;
                }
                
                /* Disabled input text color */
                input:disabled {
                    color: #94a3b8 !important;
                    -webkit-text-fill-color: #94a3b8 !important;
                }
                
                /* Labels for Text input fields */
                label[data-testid="stWidgetLabel"] > div > p {
                    color: #34d399 !important;
                    font-weight: 600 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                patient_name = st.text_input("Patient Name", placeholder="Enter patient name")

            with col2:
                patient_id = st.text_input(
                    "Patient ID",
                    value=st.session_state.patient_id,
                    disabled=True
                )

            with col3:
                patient_age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=30,
                    step=1
                )
            
            # Determine cancer stage based on prediction and confidence
            if pred_name == "normal":
                stage = "No Cancer Detected"
            elif pred_name == "benign":
                stage = "Benign Tumor (Non-cancerous)"
            else:
                if confidence > 90:
                    stage = "Stage III – Advanced Malignant"
                elif confidence > 75:
                    stage = "Stage II – Moderate Malignant"
                else:
                    stage = "Stage I – Early Malignant"

            st.success(f"Predicted Cancer Stage: **{stage}**")
            
            if not patient_name.strip():
                st.warning("⚠️ Please enter the patient name before saving the record.")
            else:
                # Save patient history
                save_patient_history(
                    patient_name,
                    patient_id,
                    pred_name.upper(),
                    confidence,
                    stage,
                    risk_score
                )
                
                st.success("✅ Patient record saved successfully!")

            st.markdown("---")
            st.markdown("📝 Doctor's Notes")
            
            # Initialize session state for notes if it doesn't exist
            if "doctor_notes_text" not in st.session_state:
                st.session_state.doctor_notes_text = ""
                
            def save_notes():
                st.session_state.doctor_notes_text = st.session_state.doctor_notes_input
                
            def clear_notes():
                st.session_state.doctor_notes_text = ""
                st.session_state.doctor_notes_input = ""
                
            # We use a form to hold the text area and the save button
            with st.form(key="doctor_notes_form"):
                st.text_area(
                    "", 
                    value=st.session_state.doctor_notes_text,
                    height=150, 
                    key="doctor_notes_input",
                    placeholder="Write any specific notes, observations, or follow-up recommendations for the patient here. These will be included in the downloaded PDF report."
                )
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    save_clicked = st.form_submit_button("Save", on_click=save_notes, type="primary", use_container_width=True)
                with col_btn2:
                    clear_clicked = st.form_submit_button("Clear", on_click=clear_notes, type="primary", use_container_width=True)
                    
                if save_clicked:
                    st.success("Notes saved for the report!")

            # The report generation will use whatever is saved in the session state
            saved_notes = st.session_state.doctor_notes_text

            # Generate and display download button for report
            # Using us_pred_name, us_confidence, histo_pred_name, histo_confidence
            # Note: These variables will only be correctly defined if scan_type == "Patient Data" 
            # due to code block inside `if scan_type == "Patient Data":` above
            report_buffer = generate_patient_report(
                patient_name,
                patient_id,
                patient_age,
                pred_name,
                confidence,
                stage,
                probs,
                us_pred=us_pred_name,
                us_conf=us_confidence,
                histo_pred=histo_pred_name,
                histo_conf=histo_confidence,
                doctor_notes=saved_notes
            )

            st.download_button(
                label="⬇️ Download Patient Report (PDF)",
                data=report_buffer,
                file_name=f"{patient_name.replace(' ', '_')}_Report.pdf" if patient_name.strip() else "Breast_Cancer_Report.pdf",
                mime="application/pdf"
            )
            
            # ========= DETAILED METRICS =========
            st.markdown('<div class="section-header">📋 Detailed Metrics</div>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            
            with cols[0]:
                st.markdown(
                    """
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 2rem; color: #60a5fa;">🔄</div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Model Architecture</div>
                        <div style="color: #cbd5e1;">ResNet18 with Transfer Learning</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with cols[1]:
                st.markdown(
                    """
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 2rem; color: #34d399;">🎯</div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Explainability</div>
                        <div style="color: #cbd5e1;">Grad-CAM Visualization</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with cols[2]:
                st.markdown(
                    """
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 2rem; color: #fbbf24;">📊</div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Confidence Scores</div>
                        <div style="color: #cbd5e1;">Real-time Probability Distribution</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

else:
    # Landing page when no image uploaded
    st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="glass-card equal-card">
                <h3> How to Use</h3>
                <ol style="color: #cbd5e1;">
                    <li>Upload a breast ultrasound image</li>
                    <li>Wait for AI analysis (2-3 seconds)</li>
                    <li>Review the prediction results</li>
                    <li>Explore visual explanations</li>
                    <li>Check confidence scores</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card equal-card">
                <h3>Features</h3>
                <ul style="color: #cbd5e1;">
                    <li>AI-powered classification</li>
                    <li>Grad-CAM heatmap visualization</li>
                    <li>Real-time confidence scoring</li>
                    <li>Interactive probability charts</li>
                    <li>Professional medical UI</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # Feature highlights
    st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)
    
    features = st.columns(4)
    
    features[0].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">🤖</div>
            <div style="font-weight: 600; margin: 10px 0;">AI-Powered</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Deep Learning Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    features[1].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">👁️</div>
            <div style="font-weight: 600; margin: 10px 0;">Explainable</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Visual Interpretability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    features[2].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">⚡</div>
            <div style="font-weight: 600; margin: 10px 0;">Real-time</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Instant Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    features[3].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">🛡️</div>
            <div style="font-weight: 600; margin: 10px 0;">Secure</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Privacy Focused</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Upload prompt
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; padding: 40px;">'
        '<h3 style="color: #22c55e;">📤 Ready to Analyze?</h3>'
        '<p style="color: #cbd5e1;">Upload your first image using the uploader above</p>'
        '</div>',
        unsafe_allow_html=True
    )

# ================= PATIENT HISTORY (CLICK TO VIEW) =================

with st.expander("📁 Patient History", expanded=False):
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No patient records available yet.")
