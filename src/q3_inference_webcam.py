import streamlit as st 
import os
import cv2
from src.DERT import *

def inference_webcam():
    st.image("image/infer_webcam.png")  
    if st.button("Runing Inference"):
        image_placeholder = st.empty()
        detr = load_model() 
        cap = cv2.VideoCapture(0)
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print('End.')
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            prob, boxes = detect(im, detr, transform) 

            frame_imshow = frame.copy()  # Copy the frame for displaying
            
            for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
                cv2.rectangle(frame_imshow, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=3)
                
                cl = p.argmax()
                text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                cv2.putText(frame_imshow, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

            # Comment the following line if you don't need to display the video
            image_placeholder.image(frame_imshow, channels="RGB")

        cap.release()