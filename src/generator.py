import os
import huggingface_hub
from pathlib import Path

import warnings
import logging

# --- TERMINAL CLEANUP ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# -------------------------

# --- MONKEY PATCHES ---
# Need to patch BEFORE imports to avoid startup crashes
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

# Disable incompatible DML convolutions cache
os.environ["DIR_ML_DISABLE_CONVOLUTION_CACHE"] = "1"
# --- END patches ---

import torch
import torch_directml
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor
from controlnet_aux import MLSDdetector
from PIL import Image
import numpy as np

# Global variables to keep the pipeline in cache after the first load
global_pipe = None
global_mlsd = None

def get_models(device):
    """Loads and caches models on the device."""
    global global_pipe, global_mlsd
    
    if global_mlsd is None:
        print("Loading MLSD Detector...")
        global_mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        
    if global_pipe is None:
        print("Loading ControlNet Model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", 
            torch_dtype=torch.float16
        )
        print("Loading Stable Diffusion...")
        global_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        print("Loading Interior Design LoRA...")
        global_pipe.load_lora_weights("data/interior_design.safetensors")
        global_pipe.fuse_lora(lora_scale=0.8) 

        global_pipe.scheduler = UniPCMultistepScheduler.from_config(global_pipe.scheduler.config)
        global_pipe.unet.set_attn_processor(AttnProcessor())
        global_pipe.to(device)
        global_pipe.enable_attention_slicing()
        
    return global_mlsd, global_pipe

def process_image(input_img_pil, user_prompt, num_steps=30, guidance=8.0, ctrl_scale=0.85):
    """Main generation function called by Gradio."""
    # Convert PIL input (numpy array) to standard PIL Image
    input_image = Image.fromarray(input_img_pil).convert("RGB")
    
    # AMD Configuration
    device = torch_directml.device()
    print(f"Running on: {device}")
    mlsd_detector, pipe = get_models(device)

    # Phase 1: Spatial Analysis (MLSD)
    print("Extracting geometric skeleton...")
    # Using suggested thresholds for cleaner windows
    skeleton_img = mlsd_detector(input_image, thr_v=0.1, thr_d=0.1)
    
    # Phase 2: Hyperparameter Tuning & Prompt Engineering
    # We add architectural photorealism tags by default
    engineered_prompt = f"Large bright window revealing daylight, {user_prompt}, modern spacious minimalist architectural photography, soft sunlight, highly detailed, photorealistic, 8k, wooden floor, cinematic lighting, octane render, unreal engine 5"
    
    # Improved negative prompt for architecture
    negative_prompt = "lowres, bad quality, blurry, distorted perspective, extra walls, mirror, painting, picture frame, wall lamp, overlapping furniture, cluttered, messy"
    
    # Phase 3: Generation
    print("Starting AI rendering...")
    result_image = pipe(
        engineered_prompt,
        image=skeleton_img,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_steps),        
        guidance_scale=float(guidance),            
        controlnet_conditioning_scale=float(ctrl_scale), 
    ).images[0]
    
    print("Rendering completed.")
    return result_image

def preview_skeleton(input_img_pil):
    """Generates and returns only the skeleton in a few seconds."""
    if input_img_pil is None:
        return None
        
    # Convert the image
    input_image = Image.fromarray(input_img_pil).convert("RGB")
    
    # Get the device (AMD) and load ONLY the necessary models
    device = torch_directml.device()
    mlsd_detector, _ = get_models(device)
    
    print("Extracting skeleton for quick preview...")
    # Extract the lines
    skeleton_img = mlsd_detector(input_image, thr_v=0.1, thr_d=0.1)
    
    return skeleton_img