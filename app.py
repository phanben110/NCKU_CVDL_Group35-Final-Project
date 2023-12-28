import streamlit as st
from streamlit_option_menu import option_menu
import os           
import cv2 
import numpy as np
from src.q1_inference_image import inference_image
from src.q2_inference_video import inference_video
from src.q3_inference_webcam import inference_webcam
from src.DERT import *
import warnings
warnings.filterwarnings("ignore")

# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="Group35 Final Project CVDL", page_icon="image/logo_csie2.png")
st.sidebar.image("image/logo_NCKU.jpeg", use_column_width=True)
with st.sidebar:
    selected = option_menu("Demo Final Project", ["1.Inference By Image", 
                                                  "2.Inference By Video", 
                                                  "3.Inference By Webcam"],
                           icons=['file-earmark-image-fill',
                                  'file-earmark-play-fill', 
                                  'webcam-fill'
                                  ], menu_icon="bars", default_index=0)


if selected == "1.Inference By Image":
    inference_image()

elif selected == "2.Inference By Video":
    inference_video()

elif selected == "3.Inference By Webcam":
    inference_webcam()

