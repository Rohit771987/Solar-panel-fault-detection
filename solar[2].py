import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from keras.preprocessing import image
import numpy as np
import cv2
from IPython.display import Image, display
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import tensorflow as hub
from tensorflow.keras import preprocessing
from keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import pandas as pd
import random
from keras.preprocessing.image import load_img
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import h5py
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model('SolarPanelFaultDetection.h5')

def file_selector(folder_path=r'C:\Users\Solan\OneDrive\Desktop\Rohit Ghadage\Project Dataa science\Solar panel fault detection\dataset1\test_set'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()

def file_selector(folder_path= filename):
    file = os.listdir(folder_path)
    selected_file = st.selectbox('Select a file', file)
    return os.path.join(folder_path, selected_file)

file = file_selector()

#uploaded_file = st.selectbox('Image_name', filename)

image_path = file
test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print (result)

#training_set.class_indices
if result[0][0] == 1:
    prediction = 'NoDefect'
    print('Nodefect')
    image = cv2.imread(image_path)
    cv2.putText(image, "No Defect predicted", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.waitKey(300)
    cv2.imshow("Predictions", image)
    st.image(image, caption = prediction, channels = 'RGB' )
    #cv2.waitKey(30000)
else:
    prediction = 'Defect'
    print('Defect')

    image = cv2.imread(image_path)
    cv2.putText(image, "Defect predicted", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.waitKey(300)
    cv2.imshow("Predictions", image)
    st.image(image, caption = prediction, channels = 'RGB' )
    #display(Image(filename=image_path))
    #cv2.waitKey(30000)