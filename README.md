# ✍️ Handwritten Digit Recognizer using CNN

A deep learning-based Handwritten Digit Recognition application that accurately predicts handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application provides an intuitive Streamlit interface where users can either draw digits on a digital canvas or upload handwritten digit images for real-time prediction.

---

## 📌 Features

- ✍️ Draw handwritten digits using an interactive canvas
- 📤 Upload handwritten digit images (PNG, JPG, JPEG)
- 🧠 CNN-based digit classification
- 📊 Displays prediction confidence score
- 📈 Visualizes probability distribution for all digits (0–9)
- 🎨 Clean and responsive Streamlit interface
- ⚡ Real-time predictions

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Streamlit**
- **Pillow**

---

## 🧠 Model Architecture

The model is built using a Convolutional Neural Network (CNN) consisting of:

- Convolutional Layers
- ReLU Activation
- Max Pooling Layers
- Fully Connected Dense Layers
- Softmax Output Layer

The model is trained on the **MNIST Handwritten Digit Dataset** containing over **70,000** handwritten digit images.

---

## 📂 Project Structure

```
Handwritten-Digit-Recognizer/
│
├── app.py
├── predict.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   └── digit_model.keras
│
└── images/
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/pranathi-555/Handwritten-Digit-Recognizer.git
```

### Navigate to Project

```bash
cd Handwritten-Digit-Recognizer
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📸 Application Features

- Draw handwritten digits directly on the canvas
- Upload digit images for prediction
- Automatic image preprocessing
- CNN-based inference
- Confidence score display
- Probability distribution chart
- User-friendly Streamlit interface

---

## 📊 Dataset

The model is trained using the **MNIST Handwritten Digit Dataset**, one of the most widely used benchmark datasets for image classification.

Dataset Size:

- **60,000** Training Images
- **10,000** Testing Images
- **10 Classes (Digits 0–9)**

---

## 🎯 Applications

- Optical Character Recognition (OCR)
- Digital Form Processing
- Educational AI Applications
- Intelligent Document Processing
- Deep Learning Demonstrations

---

## 🔮 Future Enhancements

- Support for multiple handwritten digits
- Real-time webcam digit recognition
- Improved CNN architecture for higher accuracy
- Mobile-friendly interface
- Model deployment on cloud platforms
- Batch image prediction

---

## 💻 Author

**Pranathi Kurapati**

B.Tech – Electronics and Communication Engineering (ECE)

GitHub: https://github.com/pranathi-555

---

## ⭐ If you found this project useful, please consider giving it a Star!
