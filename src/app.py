import gradio as gr
from generator import process_image, preview_skeleton 
import numpy as np
import os
import warnings
import logging

# --- CLEAN TERMINAL ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_WARNINGS"] = "1"
# ----------------------

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

title = "AI Interior Designer"
description = """
Upload a photo of an empty room. The AI will analyze the wall geometry and create a furnished rendering 
according to your desired style. 
**(Technical Note: The models runs locally on AMD RX 6600 hardware)**.
"""

with gr.Blocks(title=title) as demo:
    
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_file = gr.Image(
                label="Empty Room Photo", 
                type="numpy",
                sources=["upload"]  
            )

            preview_btn = gr.Button("🔍 Extract Structural Map", variant="secondary")
            
            skeleton_output = gr.Image(
                label="Detected MLSD Map", 
                interactive=False
            )
            
        with gr.Column(scale=3):
            output_image = gr.Image(
                label="Rendering", 
                interactive=False
            )

            with gr.Accordion("⚙️ Advanced Settings (Tuning)", open=False):
                steps_slider = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="Detail Quality (Inference Steps)")
                guidance_slider = gr.Slider(minimum=4.0, maximum=12.0, value=8.0, step=0.1, label="Prompt Fidelity (Guidance Scale)")
                control_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.05, label="Geometric Rigidity (ControlNet Scale)")

            input_prompt = gr.Textbox(
                label="Desired Interior Style", 
                placeholder="e.g. modern Scandinavian style with light oak wood",
                lines=3
            )
            
            generate_btn = gr.Button("Generate Rendering", variant="primary")
            
    preview_btn.click(
        fn=preview_skeleton,
        inputs=[input_file],
        outputs=[skeleton_output]
    )

    generate_btn.click(
        fn=process_image,
        inputs=[input_file, input_prompt, steps_slider, guidance_slider], 
        outputs=[output_image], 
    )

if __name__ == "__main__":
    print("Launching Gradio web application...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False) # Set share=True to generate a public link (lasts 72h)