import os
import io
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import cv2
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model ---
print("Loading model... This might take a moment.")
model = YOLO('best.pt')
print("Model loaded successfully.")

@app.route('/')
def home():
    """Serves the main index.html page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the file upload and returns the segmented image."""
    
    if 'image' not in request.files:
        return "No image file provided", 400

    file = request.files['image']

    if file.filename == '':
        return "No image selected", 400

    if file:
        try:
            # 1. Read the image file from the request
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # --- 2. Perform Segmentation (THIS IS THE FIX) ---
            # We explicitly tell the model to use the 'cpu'.
            # This prevents any GPU/CUDA-related crashes.
            results = model(image, device='cpu')
            # -----------------------------------------------

            # 3. Process the Results
            annotated_image = results[0].plot()

            # 4. Send the Image Back to the Frontend
            is_success, buffer = cv2.imencode(".jpg", annotated_image)
            if not is_success:
                return "Error encoding image", 500

            image_io = io.BytesIO(buffer)

            return send_file(image_io, mimetype='image/jpeg')

        except Exception as e:
            # If anything goes wrong, print the error to your terminal
            print(f"--- ERROR --- \n{e}\n-------------")
            return "Error processing image", 500

# This runs the app
if __name__ == '__main__':
    app.run(debug=True)