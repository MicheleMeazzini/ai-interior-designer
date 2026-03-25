import gradio as gr
from generator import process_image, preview_skeleton 
import numpy as np
import os
import warnings
import logging
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_WARNINGS"] = "1"

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

title = "AI-Interior Architect: Generazione d'Interni Predittiva"
description = """
Carica la foto di una stanza vuota. L'AI analizzerà la geometria dei muri e creerà un rendering arredato 
secondo lo stile desiderato. 
**(Nota tecnica: Il prototipo gira su hardware locale AMD RX 6600)**.
"""

# Defining the Graphic Interface (Layout)
with gr.Blocks(title=title) as demo:
    
    # Main Title
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    # Layout Column/Row
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=2):
            # 1. Image Upload
            input_file = gr.Image(
                label="Foto Stanza Vuota", 
                type="numpy",
                sources=["upload"]  
            )

            preview_btn = gr.Button("🔍 Estrai Mappa Strutturale", variant="secondary")
            
            # Nuovo box per mostrare lo skeleton
            skeleton_output = gr.Image(
                label="Mappa MLSD Rilevata", 
                interactive=False
            )
            
        # Right Column: Output
        with gr.Column(scale=3):
            output_image = gr.Image(
                label="Rendering", 
                interactive=False
            )

            # 2. Styling (Prompt)
            input_prompt = gr.Textbox(
                label="Stile d'Arredamento desiderato", 
                placeholder="es. modern Scandinavian style with light oak wood",
                lines=3
            )
            
            # Button (Genera)
            generate_btn = gr.Button("Genera Rendering", variant="primary")
            
    # Connecting logic to the interface (Event Handling)
    preview_btn.click(
        fn=preview_skeleton,
        inputs=[input_file],
        outputs=[skeleton_output]
    )

    generate_btn.click(
        fn=process_image, # Which function to call
        inputs=[input_file, input_prompt], # Which inputs to take
        outputs=[output_image], # Which output to update
        # Optionally, you can add gr.Progress() to show progress on the button
    )

# Starting the Web App
# You can set share=True to generate a public link (lasts 72h)
if __name__ == "__main__":
    print("Lancio applicazione web Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)