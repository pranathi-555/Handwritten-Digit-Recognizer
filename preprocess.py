import cv2
import numpy as np

def preprocess_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert if background is white
    if np.mean(gray) > 127:
        gray = 255 - gray

    # Blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Largest contour
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    digit = thresh[y:y+h, x:x+w]

    # Create square canvas
    size = max(w, h) + 40

    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # Resize
    canvas = cv2.resize(canvas, (28, 28))

    # Normalize
    canvas = canvas.astype("float32") / 255.0

    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas