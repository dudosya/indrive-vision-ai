import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import time

# =====================================================================================
# Configuration
# =====================================================================================
st.set_page_config(page_title="InVision AI by Dosbol E.", layout="wide", initial_sidebar_state="expanded")

# --- Model Paths ---
DAMAGE_MODEL_PATH = 'models/damage_model.pt'
CLEANLINESS_MODEL_PATH = 'models/cleanliness_model.pt'
SAFETY_MODEL_PATH = 'models/safety_model.pt'

# --- Cleanliness Model Config ---
IMG_SIZE = 224
CLEANLINESS_CLASSES = ['clean', 'dirty']
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- Confidence Thresholds ---
DAMAGE_CONF_THRESH = 0.40
SAFETY_CONF_THRESH = 0.50

# =====================================================================================
# Model Loading
# =====================================================================================
@st.cache_resource
def load_models():
    damage_model = YOLO(DAMAGE_MODEL_PATH)
    safety_model = YOLO(SAFETY_MODEL_PATH)
    cleanliness_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(CLEANLINESS_CLASSES))
    cleanliness_model.load_state_dict(torch.load(CLEANLINESS_MODEL_PATH, map_location=torch.device('cpu')))
    cleanliness_model.eval()
    return damage_model, cleanliness_model, safety_model

damage_model, cleanliness_model, safety_model = load_models()

# =====================================================================================
# Prediction & Scoring Functions
# =====================================================================================
# (These functions remain exactly the same)
def predict_cleanliness(image: Image.Image):
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = cleanliness_model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    return {name: prob.item() for name, prob in zip(CLEANLINESS_CLASSES, probabilities)}

def predict_yolo(model, image: Image.Image, confidence_threshold):
    img_np = np.array(image.convert('RGB'))
    results = model(img_np, conf=confidence_threshold)
    annotated_image_np = results[0].plot()
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB))
    return annotated_image, results[0].boxes

def calculate_condition_score(cleanliness_results, damage_results, safety_results):
    score = 100
    reasons = []
    is_safe = True
    if len(safety_results) > 0:
        is_safe = False
        score -= 60
        reasons.append(f"**-60 pts:** Critical safety issue detected.")
    dirty_prob = cleanliness_results.get('dirty', 0)
    if dirty_prob > 0.5:
        deduction = int(20 * dirty_prob)
        score -= deduction
        reasons.append(f"**-{deduction} pts:** Vehicle requires cleaning.")
    damage_deduction = 0
    for box in damage_results:
        damage_type = damage_model.names[int(box.cls[0])]
        if damage_type == 'scratch': damage_deduction += 5
        elif damage_type == 'dent': damage_deduction += 10
        elif damage_type == 'rust': damage_deduction += 20
    if damage_deduction > 0:
        score -= damage_deduction
        reasons.append(f"**-{damage_deduction} pts:** {len(damage_results)} cosmetic damages found.")
    score = max(0, score)
    return score, reasons, is_safe

# =====================================================================================
# Streamlit User Interface (FINAL, DEPRECATION-FREE VERSION)
# =====================================================================================

# --- Sidebar ---
st.sidebar.title("About InVision AI")
st.sidebar.info(
    "**InVision AI** is a proof-of-concept platform for the InDrive Hackathon. "
    "It demonstrates a powerful, multi-model pipeline for automated vehicle inspection."
)
st.sidebar.success("✅ AI Models are loaded and ready.")
st.sidebar.markdown("---")
placeholder = st.sidebar.empty()
placeholder.info("Upload an image and click 'Generate' to see performance metrics.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Made by:** Dosbol Erlan  
    **Team:** MRZL  
    *September 14, 2025*
    """
)

# --- Main Page ---
st.title("🚗 InVision AI: Vehicle Condition Report")
st.write("Upload a photo to generate a comprehensive quality and safety report for any vehicle.")
uploaded_file = st.file_uploader("Upload a front or side profile photo...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col_img, col_report = st.columns([2, 3])
    with col_img:
        st.image(image, caption="Uploaded Vehicle", width='stretch') # CORRECTED PARAMETER

    with col_report:
        st.write("")
        if st.button("Generate Vehicle Condition Report", width='stretch'): # CORRECTED PARAMETER
            with st.spinner("🔬 Performing multi-model analysis..."):
                start_time = time.time()
                cleanliness_results = predict_cleanliness(image)
                damage_annotated_img, damage_boxes = predict_yolo(damage_model, image, DAMAGE_CONF_THRESH)
                safety_annotated_img, safety_boxes = predict_yolo(safety_model, image, SAFETY_CONF_THRESH)
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.session_state.elapsed_time = elapsed_time
                score, reasons, is_safe = calculate_condition_score(cleanliness_results, damage_boxes, safety_boxes)
                st.subheader("🏁 Final Verdict")
                if not is_safe:
                    st.error("**FAIL:** Critical safety issues detected. Vehicle is not suitable for service.", icon="❌")
                elif score < 60:
                    st.warning(f"**ACTION REQUIRED:** Vehicle condition is poor (Score: {score}/100).", icon="⚠️")
                elif score < 85:
                    st.info(f"**PASS:** Vehicle is in fair condition (Score: {score}/100).", icon="👍")
                else:
                    st.success(f"**EXCELLENT:** Vehicle is in great condition (Score: {score}/100).", icon="✅")
                
                tab1, tab2, tab3 = st.tabs(["📊 Overall Score", "🛡️ Safety Details", "🧼 Cosmetic Details"])
                with tab1:
                    st.metric(label="Vehicle Condition Score", value=f"{score} / 100")
                    st.write("Score Breakdown:")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                with tab2:
                    if not is_safe:
                        for box in safety_boxes:
                            issue = safety_model.names[int(box.cls[0])].replace('_', ' ').title()
                            st.write(f"- **{issue}** detected with **{box.conf[0]:.1%}** confidence.")
                        st.image(safety_annotated_img, width='stretch') # CORRECTED PARAMETER
                    else:
                        st.write("No critical safety issues were found.")
                with tab3:
                    predicted_class = max(cleanliness_results, key=cleanliness_results.get)
                    st.write(f"**Cleanliness:** The vehicle is rated as **{predicted_class}** with {cleanliness_results[predicted_class]:.1%} confidence.")
                    st.write(f"**Cosmetic Damages:** Found **{len(damage_boxes)}** issue(s).")
                    if len(damage_boxes) > 0:
                        for i, box in enumerate(damage_boxes):
                            damage_type = damage_model.names[int(box.cls[0])].title()
                            st.write(f"  - **{damage_type}** (Confidence: {box.conf[0]:.1%})")
                        st.image(damage_annotated_img, width='stretch') # CORRECTED PARAMETER

if 'elapsed_time' in st.session_state:
    placeholder.metric(label="Total Analysis Time", value=f"{st.session_state.elapsed_time:.2f} s")