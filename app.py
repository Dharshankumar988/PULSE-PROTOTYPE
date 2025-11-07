import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import time

# -----------------------------------------------------------------
# 1. LOAD MODELS & CONFIG
# -----------------------------------------------------------------
print("Loading models...")
MODELS = {}

# **NEW**: Storing model path AND its best confidence score
MODEL_CONFIG = {
    "Brain (MRI)": {
        "file": "brain_model.pt",
        "conf": 0.25  # 25% confidence for brain tumors
    },
    "Bone Fracture (X-ray/CT)": {
        "file": "fracture_model.pt",
        "conf": 0.10  # 15% confidence (fractures can be subtle)
    },
    "Kidney (CT/MRI)": {
        "file": "kidney_model.pt",
        "conf": 0.30  # 30% confidence for kidney tumors
    }
}
AVAILABLE_MODELS = []

for name, config in MODEL_CONFIG.items():
    file = config["file"]
    if os.path.exists(file):
        MODELS[name] = YOLO(file)
        AVAILABLE_MODELS.append(name)
        print(f"Loaded model: {name} (Conf: {config['conf']})")
    else:
        print(f"WARNING: Model file not found: {file}. Skipping {name}.")
print("Model loading complete.")

# -----------------------------------------------------------------
# 2. HELPER FUNCTIONS (No Change)
# -----------------------------------------------------------------

def fix_colors(cv_img: np.ndarray) -> np.ndarray:
    """Converts OpenCV BGR (from .plot()) to Gradio's expected RGB format."""
    if cv_img is None: return None
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

def results_to_table(results):
    """Converts a YOLO results object to a list for a gr.DataFrame."""
    rows = []
    if not results or not results[0].boxes:
        return rows
        
    names = results[0].names or {}
    for b in results[0].boxes:
        cls = int(b.cls)
        conf = float(b.conf)
        xyxy = b.xyxy[0].tolist()
        name = names.get(cls, f"Class {cls}")
        rows.append([name, round(conf, 3), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
    return rows

# -----------------------------------------------------------------
# 3. CORE INFERENCE FUNCTION (Updated)
# -----------------------------------------------------------------

# **UPDATED**: This function no longer needs 'conf_thresh'
def run_inference(model_key, input_image, progress=gr.Progress(track_tqdm=True)):
    """
    Main function to run analysis.
    Takes a model key and image, returns annotated image, table, and status.
    """
    # 1. Handle Inputs
    if input_image is None:
        return None, [], "Status: No image uploaded. Please upload an image."
        
    model = MODELS.get(model_key)
    if model is None:
        return input_image, [], f"Status: ERROR! Model '{model_key}' is not loaded."

    # **NEW**: Get the specific confidence for this model
    model_conf = MODEL_CONFIG[model_key]["conf"]

    # 2. Show Progress and Run Model
    progress(0, desc="Warming up...")
    time.sleep(0.1)
    
    try:
        results = model(
            input_image, 
            device='cpu', 
            conf=model_conf,  # Use the model-specific confidence
            verbose=False 
        )
    except Exception as e:
        print(f"--- INFERENCE ERROR --- \n{e}\n---------------------")
        return input_image, [], f"Status: An error occurred: {e}"

    # 3. Process and Return Outputs
    annotated_image = fix_colors(results[0].plot())
    table_data = results_to_table(results)
    
    status = f"Status: Success! Found {len(table_data)} detections at {model_conf*100}% conf."
    
    return annotated_image, table_data, status

# -----------------------------------------------------------------
# 4. DYNAMIC HEADER FUNCTION (No Change)
# -----------------------------------------------------------------

def update_header_and_theme(model_key):
    """
    Called when the dropdown changes.
    Returns a new title and new CSS for the background color.
    """
    if "Brain" in model_key:
        emoji = "🧠"
        title_html = f"<h1>{emoji} Brain Tumour Analyser</h1>"
        css = "<style>body { background-color: #f0f8ff !important; }</style>" 
    elif "Fracture" in model_key:
        emoji = "🦴"
        title_html = f"<h1>{emoji} Bone Fracture Analyser</h1>"
        css = "<style>body { background-color: #f5f5dc !important; }</style>" 
    elif "Kidney" in model_key:
        emoji = "🩸"
        title_html = f"<h1>{emoji} Kidney Tumour Analyser</h1>"
        css = "<style>body { background-color: #f0fff0 !important; }</style>"
    else:
        emoji = "🩺"
        title_html = f"<h1>{emoji} Medical Scan Analyser</h1>"
        css = "<style>body { background-color: #f4f7f9 !important; }</style>"
    
    return gr.update(value=title_html), gr.update(value=css)

# -----------------------------------------------------------------
# 5. THEME & CSS (No Change)
# -----------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_lg,
    font="Poppins, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
).set(
    body_background_fill="#f4f7f9",
    background_fill_primary="#FFFFFF",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
    border_color_primary="*neutral_200",
    body_text_color="*neutral_800",
    body_text_weight="400",
)

CUSTOM_CSS = """
h1 {
    text-align: center; display: block;
    font-size: 2.5rem !important; font-weight: 700 !important;
    color: var(--primary_500) !important; /* THIS IS THE FIX: USES THEME'S BLUE */
    margin-top: 20px; margin-bottom: 0px;
}
p.subheading {
    text-align: center; display: block;
    font-size: 1.1rem; color: #555; margin-top: 0; margin-bottom: 20px;
}
/* This correctly hides the footer */
footer { display: none !important }
"""

# -----------------------------------------------------------------
# 6. UI LAYOUT (gr.Blocks) (Updated)
# -----------------------------------------------------------------

with gr.Blocks(title="Medical Scan Analyser", theme=theme, css=CUSTOM_CSS, mode="light") as demo:
    
    dynamic_style = gr.HTML()
    
    main_title = gr.Markdown("<h1>🩺 Medical Scan Analyser</h1>")
    gr.Markdown("<p class'subheading'>Upload a scan to detect abnormalities. <b>Demo only — not for medical diagnosis.</b></p>")

    with gr.Row(variant="panel"):
        
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload & Configure")
            
            input_image = gr.Image(
                type="pil", 
                label="Upload Scan (PNG, JPG)"
            )
            
            model_selector = gr.Dropdown(
                AVAILABLE_MODELS, 
                value=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None, 
                label="Choose Analyser Model"
            )
            
            gr.Examples(
                examples=[
                    ["samples/sample_brain.jpg", "Brain (MRI)"],
                    ["samples/sample_fracture.jpg", "Bone Fracture (X-ray/CT)"],
                    ["samples/sample_kidney.png", "Kidney (CT/MRI)"],
                ],
                inputs=[input_image, model_selector], 
                label="Sample Images (Click to use)"
            )
            
            # **CONFIDENCE SLIDER REMOVED**
            
            run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 2. View Results")
            
            output_image = gr.Image(
                type="pil", 
                label="Annotated Result",
                interactive=False
            )

            status_text = gr.Textbox(
                label="Status", 
                value="Ready. Please upload an image and run analysis.", 
                interactive=False
            )
            
            detections_df = gr.DataFrame(
                headers=["Label", "Confidence", "X1", "Y1", "X2", "Y2"], 
                datatype=["str", "number", "number", "number", "number", "number"],
                label="Detections Table",
                interactive=False,
                row_count=(3, "auto"), 
                wrap=True
            )

    # -----------------------------------------------------------------
    # 7. EVENT LISTENERS (Callbacks) (Updated)
    # -----------------------------------------------------------------
    
    # **UPDATED**: 'conf_slider' is removed from the inputs list
    run_btn.click(
        fn=run_inference,
        inputs=[
            model_selector,
            input_image
        ],
        outputs=[
            output_image,
            detections_df,
            status_text
        ]
    )
    
    model_selector.change(
        fn=update_header_and_theme,
        inputs=model_selector,
        outputs=[
            main_title,
            dynamic_style
        ]
    )

# -----------------------------------------------------------------
# 8. LAUNCH THE APP (No Change)
# -----------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(debug=True, server_port=8080)