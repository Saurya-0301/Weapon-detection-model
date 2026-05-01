import streamlit as st
import cv2
from ultralytics import YOLO
import torch
import os

# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(page_title="AI Defense Surveillance", layout="wide")

st.title("🛡 AI Defense Surveillance Command Center")

# -------------------------
# LOAD MODELS
# -------------------------

device = 0 if torch.cuda.is_available() else "cpu"

general_model = YOLO("models/yolov8m.pt")
weapon_model = YOLO("models/weapon_best.pt")

# -------------------------
# DASHBOARD LAYOUT
# -------------------------

left, right = st.columns([3,1])

camera_placeholder = left.empty()

with right:
    st.subheader("System Status")
    persons_metric = st.metric("Persons Detected", 0)
    weapons_metric = st.metric("Weapons Detected", 0)
    threat_metric = st.metric("Threat Level", "LOW")

st.divider()

st.subheader("Evidence Gallery")
gallery = st.empty()

# -------------------------
# CAMERA
# -------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        st.error("Camera not found")
        break

    annotated = frame.copy()

    person_count = 0
    weapon_count = 0

    # -------------------------
    # PERSON DETECTION
    # -------------------------

    results = general_model(frame, conf=0.5, device=device)

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        label = general_model.names[cls_id]

        if label == "person":

            person_count += 1

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

    # -------------------------
    # WEAPON DETECTION
    # -------------------------

    weapon_results = weapon_model(frame, conf=0.7, device=device)

    for box in weapon_results[0].boxes:

        weapon_count += 1

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),3)

    # -------------------------
    # THREAT LOGIC
    # -------------------------

    threat = "LOW"

    if weapon_count > 0 and person_count > 0:
        threat = "HIGH"

    elif weapon_count > 0:
        threat = "MEDIUM"

    # -------------------------
    # UPDATE DASHBOARD
    # -------------------------

    camera_placeholder.image(annotated, channels="BGR")

    persons_metric.metric("Persons Detected", person_count)
    weapons_metric.metric("Weapons Detected", weapon_count)
    threat_metric.metric("Threat Level", threat)

    # -------------------------
    # EVIDENCE GALLERY
    # -------------------------

    if os.path.exists("evidence"):

        images = sorted(os.listdir("evidence"), reverse=True)[:6]

        cols = gallery.columns(3)

        for i, img in enumerate(images):

            cols[i % 3].image(f"evidence/{img}", use_container_width=True)
