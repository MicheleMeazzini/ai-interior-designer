import torch_directml
from controlnet_aux import MLSDdetector
from PIL import Image
import os

def extract_room_skeleton(image_path, output_path):
    device = torch_directml.device()
    
    input_image = Image.open(image_path).convert("RGB")

    print("Loading MLSD model...")
    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    
    print("Extracting architectural lines...")
    detected_map = mlsd(input_image, thr_v=1, thr_d=1)
    
    detected_map.save(output_path)
    print(f"Map successfully saved to: {output_path}")
