🧠 Pulse AI Prototype 🦴
A multi-task deep learning web application built with Gradio and YOLOv8. This tool provides a tabbed interface for analysing different types of medical scans to detect abnormalities.

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
Install Requirements: Make sure your requirements.txt file contains:

gradio
ultralytics
opencv-python-headless
numpy
Then run:

Bash

pip install -r requirements.txt
Run the App:

Bash

python app.py
The app will now be running on a local URL like http://127.0.0.1:7860.
☁️ How to Deploy on Hugging Face
This project is designed to be deployed on Hugging Face Spaces for free.

Push to GitHub: Ensure your repository is up-to-date with all the files listed in the project structure (especially the .pt models and requirements.txt).

Create a Hugging Face Account:

Log in to HuggingFace.co (preferably by linking your GitHub account).

Create a New Space:

Click your profile icon → New Space.

Space name: Pulse-AI-Prototype

License: MIT

Space SDK: Select Gradio (This is the most important step).

Hardware: Leave it on CPU basic - Free. This provides 16GB of RAM, which is enough for your models.

Click Create Space.

Import from GitHub:

On your new Space page, click the "Files" tab.

Click "Add file" → "Import from GitHub".

Repo URL: Paste the URL of your GitHub repository.

Branch: main (or master).

Click "Import".

Hugging Face will automatically pull your code, install the libraries from requirements.txt, and start your app.py service. The initial build may take 5-10 minutes. Once "Running", your app will be live for anyone to use.