import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload File", type=["jpg", "png", "jpeg"])

conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
show_labels = st.checkbox("Show Labels", True)
box_color = st.color_picker("Box Color", "#00FF00")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image")

    results = model(image)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # convert hex color → BGR
    hex_color = box_color.lstrip("#")
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    for box in results[0].boxes:

        if box.conf[0] < conf_thres:
            continue

        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls = int(box.cls[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2)

        if show_labels:
            label = model.names[cls]
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)

    st.image(img, channels="BGR")