import torch_directml
from controlnet_aux import MLSDdetector
from PIL import Image
import os

def extract_room_skeleton(image_path, output_path):
    # 1. Device Configuration (AMD)
    device = torch_directml.device()
    
    # 2. Image Loading
    # GenAI works best with square images or dimensions that are multiples of 64
    input_image = Image.open(image_path).convert("RGB")
    
    # 3. MLSD Detector Initialization
    # MLSD = Mobile Line Segment Detection (excellent for interiors)
    print("Loading MLSD model...")
    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    
    # 4. Processing
    # thr_v and thr_d are the detection thresholds for value and distance
    print("Extracting architectural lines...")
    detected_map = mlsd(input_image, thr_v=0.1, thr_d=0.1)
    
    # 5. Saving
    detected_map.save(output_path)
    print(f"Map successfully saved to: {output_path}")

if __name__ == "__main__":
    # Make sure you have a photo named 'room.jpg' in the folder
    # or change the name below
    test_image = "data/input/room.jpg" 
    if os.path.exists(test_image):
        extract_room_skeleton(test_image, "data/output/room_skeleton.png")
    else:
        print(f"Error: Please upload a photo named '{test_image}' in the folder!")