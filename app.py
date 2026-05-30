import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import time

# -----------------------------------------------------------------
# 1. LOAD MODELS & CONFIG (No Change)
# -----------------------------------------------------------------
print("Loading models...")
MODELS = {}

MODEL_CONFIG = {
    "Brain (MRI)": {
        "file": "brain_model.pt",
        "conf": 0.25
    },
    "Bone Fracture (X-ray/CT)": {
        "file": "fracture_model.pt",
        "conf": 0.10
    },
    "Kidney (CT/MRI)": {
        "file": "kidney_model.pt",
        "conf": 0.30
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
    if cv_img is None: return None
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

def results_to_table(results):
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
# 3. CORE INFERENCE FUNCTION (No Change)
# -----------------------------------------------------------------

def run_inference(model_key, input_image, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        return None, [], "Status: No image uploaded. Please upload an image."
    model = MODELS.get(model_key)
    if model is None:
        return input_image, [], f"Status: ERROR! Model '{model_key}' is not loaded."
    model_conf = MODEL_CONFIG[model_key]["conf"]
    progress(0, desc="Warming up...")
    time.sleep(0.1)
    try:
        results = model(input_image, device='cpu', conf=model_conf, verbose=False)
    except Exception as e:
        print(f"--- INFERENCE ERROR --- \n{e}\n---------------------")
        return input_image, [], f"Status: An error occurred: {e}"
    annotated_image = fix_colors(results[0].plot())
    table_data = results_to_table(results)
    status = f"Status: Success! Found {len(table_data)} detections at {model_conf*100}% conf."
    return annotated_image, table_data, status

# -----------------------------------------------------------------
# 4. DYNAMIC HEADER FUNCTION (No Change)
# -----------------------------------------------------------------
def update_header_and_theme(model_key):
    if "Brain" in model_key:
        emoji = "🧠"
        title_html = f"<h1>{emoji} Brain Tumour Analyser</h1>"
    elif "Fracture" in model_key:
        emoji = "🦴"
        title_html = f"<h1>{emoji} Bone Fracture Analyser</h1>"
    elif "Kidney" in model_key:
        emoji = "🩸"
        title_html = f"<h1>{emoji} Kidney Tumour Analyser</h1>"
    else:
        emoji = "🩺"
        title_html = f"<h1>{emoji} PULSE AI prototype</h1>"
    return gr.update(value=title_html), gr.update(value="")

# -----------------------------------------------------------------
# 5. THEME, CSS, & JAVASCRIPT
# -----------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_lg,
    font="Poppins, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)

# This is the CSS for your dark mode theme
CUSTOM_CSS = """
/* ---- DARK MODE CONTAINER ---- */
.gradio-container {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
/* ---- HEADERS ---- */
h1 {
    text-align: center;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #58a6ff !important;
    margin-top: 12px;
}
p.subheading {
    text-align: center;
    color: #c9d1d9 !important;
}
h3 {
    color: #e6edf3 !important;
}
.gr-label {
    color: #c9d1d9 !important;
}
/* ---- BUTTONS ---- */
button.primary {
    background: linear-gradient(90deg, #2384ff, #1f6feb) !important;
    color: white !important;
    border: none !important;
}
button.primary:hover {
    background: linear-gradient(90deg, #1f6feb, #1557b0) !important;
}
button.secondary {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
}
button.secondary:hover {
    background: #30363d !important;
}
/* ---- INPUTS ---- */
.upload-container, .gr-select-input, .gr-textbox {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    border: 1px solid #30363d !important;
}
/* ---- DATAFRAME & PANELS ---- */
.dataframe, .gradio-container .panel, .gradio-row.panel {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
}
footer { display: none !important; }
/* ---- IMAGE FOOTER ICONS ---- */
.gradio-container .gr-file-preview-footer svg,
.gradio-container .image .image-footer svg {
    fill: #ffffff !important;
    opacity: 0.95 !important;
    transition: opacity 0.12s ease-out, transform 0.12s ease-out;
}
.gradio-container .gr-file-preview-footer button:hover svg,
.gradio-container .image .image-footer button:hover svg {
    opacity: 1 !important;
    transform: scale(1.14);
}
.gradio-container .gr-file-preview-footer,
.gradio-container .image .image-footer {
    background: #0b1216 !important;
    border-top: 1px solid rgba(255,255,255,0.03) !important;
}
.webcam-container {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
}
"""

# This is the JavaScript to hide the DataFrame menu items
PRUNE_MENU_SCRIPT = """
<script>
(function() {
  function pruneMenu(node) {
    if(!node) return;
    try {
      const items = node.querySelectorAll('div, li, button, span');
      items.forEach(it => {
        const text = (it.innerText || '').trim().toLowerCase();
        if (!text) return;
        const unwanted = [
          'clear sort', 'clear sorts', 'clear filters', 'filter',
          'remove sort', 'insert left', 'insert right', 'hide column',
          'remove column', 'group', 'ungroup'
        ];
        if (text.includes('sort ascending') || text.includes('sort descending')) {
          return; /* Keep these */
        }
        for (let u of unwanted) {
          if (text.includes(u)) {
            it.style.display = 'none';
            return;
          }
        }
      });
    } catch(e) { console.log('pruneMenu error', e); }
  }
  function scanExistingMenus(){
    document.querySelectorAll('.gr-svelte-select-menu, .dataframe-menu, [role=\"menu\"]').forEach(pruneMenu);
  }
  window.addEventListener('load', () => {
    scanExistingMenus();
    const mo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        m.addedNodes.forEach(node => {
          if (node.nodeType === 1) { /* is HTMLElement */
             if (node.matches('.gr-svelte-select-menu, .dataframe-menu, [role=\"menu\"]')) {
                pruneMenu(node);
             } else {
                node.querySelectorAll('.gr-svelte-select-menu, .dataframe-menu, [role=\"menu\"]').forEach(pruneMenu);
             }
          }
        });
      }
    });
    mo.observe(document.body, { childList: true, subtree: true });
  });
})();
</script>
"""

# **THE FIX**: We combine CSS and JS into one string for the 'head'
APP_HEAD = f"""
<style>
{CUSTOM_CSS}
</style>

{PRUNE_MENU_SCRIPT}
"""

# -----------------------------------------------------------------
# 6. UI LAYOUT (gr.Blocks)
# -----------------------------------------------------------------

# **THE FIX**: We use the 'head' argument to inject everything.
# We also remove `css=...` and `mode="light"`.
with gr.Blocks(title="PULSE AI prototype", theme=theme, head=APP_HEAD) as demo:
    
    dynamic_style = gr.HTML() # Still needed as a target for the model_selector
    
    main_title = gr.Markdown("<h1>🩺 PULSE AI prototype</h1>")
    gr.Markdown("<p class='subheading'>Upload a scan to detect abnormalities. <b>Demo only — not for medical diagnosis.</b></p>")

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
    # 7. EVENT LISTENERS (Callbacks) (No Change)
    # -----------------------------------------------------------------
    
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        debug=False,
    )