🧠 PULSE AI prototype 🦴
A multi-task deep learning web application built with Gradio and YOLOv8. It runs as a Hugging Face Spaces web app and lets users upload scans in the browser to detect abnormalities.

Tab 1: Brain Tumor (MRI) Analyser

Tab 2: Bone Fracture (CT/X-ray) Analyser

Tab 3: Kidney Tumor (CT/MRI) Analyser

⚠️ CRITICAL MEDICAL DISCLAIMER

This project is for educational and demonstration purposes only. The AI models were trained on public datasets and are NOT a substitute for professional medical advice, diagnosis, or treatment. Never disregard professional medical advice because of something you have seen on this application.

✨ Features
Multi-Model Interface: Uses a clean, tabbed interface from Gradio to serve three independent AI models.

AI-Powered Detection: Employs three separate YOLOv8 detection models (.pt files) trained on custom data.

Visual Feedback: The app takes an image as input and returns the same image with bounding boxes drawn around any detected abnormalities.

Simple & Fast: Built with Gradio for a fast, responsive, and easy-to-use UI.

Hugging Face Ready: Designed specifically for easy, free deployment on Hugging Face Spaces.

💻 Tech Stack
Backend & Interface: Gradio

AI Models: YOLOv8 (from Ultralytics / PyTorch)

Image Processing: OpenCV & NumPy

Deployment: Hugging Face Spaces

📁 Final Project Structure
This project is configured for a standard Hugging Face Spaces deployment.

PULSE-PROTOTYPE/
├── app.py             # The main Gradio application
├── brain_model.pt     # Trained YOLOv8 model for brain tumors
├── fracture_model.pt  # Trained YOLOv8 model for bone fractures
├── kidney_model.pt    # Trained YOLOv8 model for kidney tumors
├── requirements.txt   # A list of all Python dependencies
└── README.md          # This file
🚀 How to Run Locally
Clone the Repository:

Bash

git clone https://github.com/Dharshankumar988/PULSE-PROTOTYPE.git
cd PULSE-PROTOTYPE
(Optional) Create a Virtual Environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
Install Requirements:

Bash

pip install -r requirements.txt

Then run:

Bash

python app.py
The app will now be running on a local URL like http://127.0.0.1:7860.
☁️ How to Deploy on Hugging Face
1. Push this repository to GitHub with `master` up to date.
2. Go to Hugging Face and create a new Space.
3. Set the SDK to `Gradio`.
4. Choose `CPU basic` for the hardware.
5. Import this GitHub repo into the Space or connect the repo directly.
6. Wait for the first build to finish.

Once the Space is running, Hugging Face gives you a public web URL. Open that URL in any browser and use the app like a normal web page: upload a scan, choose a model, and click `Run Analysis`.

### Notes for Spaces
- `app.py` now binds to `0.0.0.0` and uses the Space port automatically.
- `opencv-python-headless` is used so the app works cleanly in a server environment.
- Make sure the `.pt` model files stay in the repository or are stored with Git LFS.