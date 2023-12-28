import streamlit as st 
import os
import cv2
import numpy as np
from PIL import Image
from src.DERT import *

import cv2
import numpy as np
import argparse

def infer(source):  
    detr = load_model() 
    cap = cv2.VideoCapture(source)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('End.')
            break
        
        frame_imshow = frame.copy() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        prob, boxes = detect(im, detr, transform) 

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cv2.rectangle(frame_imshow, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=3)
            
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            cv2.putText(frame_imshow, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        cv2.imshow('Inference DERT', frame_imshow) 

        key = cv2.waitKey(1)  # Add a slight delay (e.g., 30 milliseconds)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical Flow Tracking')
    parser.add_argument('--path_video', type=str, default="", help='Path to the input video file')
    parser.add_argument('--camera', type=int, default=0, help='Maximum number of corners')
    args = parser.parse_args()

    if args.path_video == "": 
        infer(args.camera)
    else: 
        infer(args.path_video)
