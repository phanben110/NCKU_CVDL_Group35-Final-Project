import streamlit as st 
import os
import cv2
import numpy as np
from src.DERT import *

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    st.pyplot(plt)

def inference_image(): 
    st.image("image/infer_img.png")

    st.sidebar.header("Load Image For Inference")
    img_infer = np.array([])

    uploaded_file = st.sidebar.file_uploader("Load Image inference", type=["png","jpg"], accept_multiple_files=False)
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img_infer = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img_infer is not None:
            img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
            st.sidebar.image(img_infer, caption="Image for inference", use_column_width=True)

        if st.button("Runing Inference"):
            if img_infer is not None:
                im = Image.fromarray(img_infer)
                detr = load_model() 
                scores, boxes = detect(im, detr, transform) 
                plot_results(im, scores, boxes) 

            else:
                st.sidebar.error(f"Failed to read image: {uploaded_file.name}")