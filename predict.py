import numpy as np
import cv2
from preprocess import preprocess_image

import pickle

# Load model safely using streamlit cache
import streamlit as st

@st.cache_resource
def load_model_file():
    import tensorflow as tf
    return tf.keras.models.load_model("model/digit_model.keras")

model = load_model_file()

def predict_digit(image):

    processed = preprocess_image(image)

    if processed is None:
        return None, None, None

    prediction = model.predict(processed, verbose=0)[0]

    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100

    return digit, confidence, prediction