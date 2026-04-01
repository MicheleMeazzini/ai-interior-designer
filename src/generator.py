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
import gc
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor
from controlnet_aux import MLSDdetector, MidasDetector
from PIL import Image
import numpy as np

# Global variables to keep the pipeline in cache after the first load
global_pipe = None
global_mlsd = None
global_depth_estimator = None

depthmap_weight = 0.25
ctrl_scale = 0.9

def get_models(device):
    """Loads and caches models on the device."""

    global global_pipe, global_mlsd, global_depth_estimator

    if global_mlsd is None:
        print("Loading MLSD Detector...")
        global_mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        
    if global_depth_estimator is None:
        print("Loading Depth Estimator (MiDaS)...")
        global_depth_estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")

    if global_pipe is None:
        print("Loading ControlNet Model...")
        controlnet_mlsd = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)
        controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)

        print("Loading Stable Diffusion...")
        global_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[controlnet_mlsd, controlnet_depth],
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        print("Loading Interior Design LoRA...")
        global_pipe.load_lora_weights("data/interior_design.safetensors")
        global_pipe.fuse_lora(lora_scale=0.7) 

        global_pipe.scheduler = UniPCMultistepScheduler.from_config(global_pipe.scheduler.config)
        
        # Spostiamo tutto sulla scheda video AMD
        global_pipe.to(device)
        
        print("Enabling Memory Optimizations...")
        global_pipe.enable_attention_slicing("max")
        global_pipe.enable_vae_slicing()
        
    return global_mlsd, global_depth_estimator, global_pipe

def process_image(input_img_pil, user_prompt, num_steps=30, guidance=8.0):
    """Main generation function called by Gradio."""

    global depthmap_weight, ctrl_scale

    gc.collect()
    torch_directml.device()

    # Convert PIL input (numpy array) to standard PIL Image
    input_image = Image.fromarray(input_img_pil).convert("RGB")

    max_size = 512.0
    ratio = max_size / max(input_image.size)
    new_w = int((input_image.size[0] * ratio) // 8 * 8) 
    new_h = int((input_image.size[1] * ratio) // 8 * 8)
    
    input_image = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # AMD Configuration
    device = torch_directml.device()
    print(f"Running on: {device}")
    mlsd_detector, depth_estimator, pipe = get_models(device)

    # Spatial Analysis
    print("Extracting geometric skeleton and depth map...")
    skeleton_img = mlsd_detector(input_image, thr_v=0.1, thr_d=0.1)
    depth_map = depth_estimator(input_image)

    # We add architectural photorealism tags by default
    engineered_prompt = f"{user_prompt}, highly detailed, photorealistic, 8k, realistic lighting, octane render, unreal engine 5"
    negative_prompt = "lowres, bad quality, blurry, distorted perspective, extra walls, mirror, painting, picture frame, wall lamp, overlapping furniture, cluttered, messy,mutated furniture, asymmetrical architecture, floating objects, merged geometry, nonsensical shapes, Escher-like, abstract furniture, broken physics, missing table legs, deformed"
    
    # Phase 3: Generation
    print("Starting AI rendering...")
    result_image = pipe(
        engineered_prompt,
        image=[skeleton_img, depth_map],
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_steps),        
        guidance_scale=float(guidance),            
        controlnet_conditioning_scale=[ctrl_scale, depthmap_weight], 
        control_guidance_end=[1.0, 0.4],
        width=new_w,
        height=new_h,
    ).images[0]
    
    print("Rendering completed.")
    return result_image

def preview_skeleton(input_img_pil):
    """Generates and returns only the skeleton in a few seconds."""

    gc.collect()
    torch_directml.device()

    if input_img_pil is None:
        return None
        
    # Convert the image
    input_image = Image.fromarray(input_img_pil).convert("RGB")
    
    # Get the device (AMD) and load ONLY the necessary models
    device = torch_directml.device()
    mlsd_detector, _, _ = get_models(device)
    
    print("Extracting skeleton for quick preview...")
    # Extract the lines
    skeleton_img = mlsd_detector(input_image, thr_v=0.1, thr_d=0.1)
    
    return skeleton_img