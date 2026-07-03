import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from predict import predict_digit

# ---------------- Helper Functions ---------------- #

def show_prediction(probabilities):

    probs = probabilities * 100

    df = pd.DataFrame({

        "Digit": list(range(10)),

        "Confidence (%)": probs

    })

    st.markdown("---")

    st.subheader("📈 Prediction Probability")

    st.bar_chart(
        df.set_index("Digit"),
        use_container_width=True
    )

    top3 = df.sort_values(
        "Confidence (%)",
        ascending=False
    ).head(3)

    st.markdown("---")

    st.subheader("🏆 Top 3 Predictions")

    medals = ["🥇","🥈","🥉"]

    for i, (_, row) in enumerate(top3.iterrows()):

        st.write(
            f"{medals[i]} Digit **{int(row['Digit'])}** — **{row['Confidence (%)']:.2f}%**"
        )

# ---------------- Page Configuration ---------------- #
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🧠",
    layout="wide"
)

# ---------------- Sidebar ---------------- #
st.sidebar.title("🧠 Model Information")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📌 Project

CNN Handwritten Digit Recognizer

---

### 🤖 Deep Learning Model

Convolutional Neural Network (CNN)

---

### 📂 Dataset

MNIST Dataset

- 60,000 Training Images
- 10,000 Test Images

---

### 🧠 Technologies

- TensorFlow
- Keras
- OpenCV
- Streamlit
- NumPy

---

### 🎯 Classes

Digits **0 – 9**

---

### 📏 Input Size

28 × 28 Pixels

---

### 🚀 Version

Version 2.0
""")
# ---------------- Header ---------------- #

st.title("🧠 AI Handwritten Digit Recognizer")

st.markdown("""
### Deep Learning based Handwritten Digit Classification

Recognize handwritten digits using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

The application supports both **drawing digits directly** and **uploading handwritten images** for prediction.
""")

st.markdown("---")

st.sidebar.success("✅ CNN Model Loaded Successfully")
st.sidebar.info(
"""
💡 Tip

Draw one digit at a time in the center of the canvas for best accuracy.
"""
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Model Accuracy",
        value="99.3%"
    )

with col2:
    st.metric(
        label="📂 Dataset",
        value="MNIST"
    )

with col3:
    st.metric(
        label="🧠 Classes",
        value="10"
    )

with col4:
    st.metric(
        label="⚡ Framework",
        value="TensorFlow"
    )

st.markdown("---")

tab1, tab2 = st.tabs([
    "✏️ Draw Digit",
    "🖼️ Upload Image"
])
# ===========================
# DRAW DIGIT TAB
# ===========================

with tab1:

    st.info(
"""
✍️ Draw a **single digit (0–9)** in the center of the canvas.

For best accuracy:

• Draw only one digit

• Use thick strokes

• Avoid touching the edges
"""
)

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🔍 Predict Drawing"):

     if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        digit, confidence, probabilities = predict_digit(img)

        if digit is None:
            st.error("No digit detected!")

        else:

            col1, col2 = st.columns([1,1])

            with col1:
                st.success("Prediction Completed Successfully!")

                st.metric(
                    label="🎯 Predicted Digit",
                    value=str(digit)
                )

                st.metric(
                    label="📊 Confidence",
                    value=f"{confidence:.2f}%"
                )

            with col2:
                st.info("""
### 🧠 Model Details

✔ CNN Model

✔ MNIST Dataset

✔ TensorFlow + OpenCV
""")

            show_prediction(probabilities) 
 # ===========================
# IMAGE UPLOAD TAB
# ===========================

with tab2:

    st.info("""
📤 Upload a clear handwritten digit.

Supported formats:

• PNG

• JPG

• JPEG
""")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )

        image_np = np.array(image)

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        digit, confidence, probabilities = predict_digit(image_bgr)

        if digit is None:

            st.error("No digit detected!")

        else:

            col1, col2 = st.columns([1,1])

            with col1:

                st.success("Prediction Completed Successfully!")

                st.metric(
                    label="🎯 Predicted Digit",
                    value=str(digit)
                )

                st.metric(
                    label="📊 Confidence",
                    value=f"{confidence:.2f}%"
                )

            with col2:

                st.info("""
### 🧠 Model Details

✔ CNN Model

✔ MNIST Dataset

✔ 10 Output Classes

✔ TensorFlow + OpenCV
""")

            show_prediction(probabilities)
st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

### 🚀 Built with

TensorFlow • OpenCV • Streamlit • NumPy

---

Developed by **Pranathi Kurapati**

</div>
""",
unsafe_allow_html=True
)