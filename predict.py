import numpy as np
import cv2
from preprocess import preprocess_image
from tensorflow.keras.models import load_model

model = load_model("model/digit_model.keras")

def predict_digit(image):

    processed = preprocess_image(image)

    if processed is None:
        return None, None, None

    prediction = model.predict(processed, verbose=0)[0]

    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100

    return digit, confidence, prediction