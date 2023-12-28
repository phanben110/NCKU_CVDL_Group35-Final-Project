import streamlit as st 
import os 
import tempfile
import cv2
import numpy as np
from src.DERT import *

def inference_video(): 
    st.image("image/infer_video.png")
    # Create a sidebar for video upload
    st.sidebar.header("Upload Video", divider='rainbow') 
    uploaded_file = st.sidebar.file_uploader("",type=["mp4", "mov", "avi"])

    #max_conner is int min = 0 and max = 10 

    if uploaded_file is None: 
        st.warning("Please upload a video file.", icon="⚠️" ) 
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read()) 

        if st.button("Runing Inference"):
            image_placeholder = st.empty()
            detr = load_model() 
            cap = cv2.VideoCapture(tfile.name)
            
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

            st.success('Run scrip successful !', icon="✅") 
