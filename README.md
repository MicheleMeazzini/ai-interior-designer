# 🏡 AI Interior Designer

A Generative AI prototype for real estate and interior design. This tool takes a photo of an empty room, analyzes its geometry, and instantly generates a photorealistic, fully furnished rendering based on a text prompt.

## 📸 Before & After

| Empty Room (Input) | Room Skeleton | Furnished Room (Output) |
| :---: | :---: | :---: |
| <img src="docs/before.jpg" width="250"/> | <img src="docs/skeleton.jpg" width="250"/> | <img src="docs/after.jpg" width="250"/> |

*Example Prompt*: "A bright and airy Scandinavian living room interior, filled with abundant natural daylight streaming through large windows. Light oak hardwood floor covered by a soft textured wool rug. A comfortable grey fabric sofa with clean lines, paired with a warm brown leather armchair and a minimalist pale wood coffee table. Decorative elements include a chunky knit throw blanket, potted green plants (Monstera, Fiddle Leaf Fig), minimalist abstract wall art, stacked design books, and simple ceramic vases. High detailed, photorealistic, 8k resolution, architectural photography, soft natural light, volumetric daylight, cinematic lighting, hygge aesthetic"

## ✨ Key Features
* **Structural Preservation:** Uses ControlNet (MLSD) to detect and preserve the original walls, floors, and windows.
* **Specialized Fine-Tuning:** Integrates LoRA weights to improve photorealism, lighting, and textures for architectural visualization.
* **Custom Styles:** Type any interior design style (Modern, Industrial, Victorian, etc.) to dynamically change the room's look.
* **Local Execution:** Optimized to run entirely on local hardware ensuring data privacy (tested on AMD RX 6600).
* **Simple UI:** A clean, browser-based interface powered by Gradio.

## 🚀 Quick Start

This project uses `uv` for lightning-fast dependency management.

1. Clone the repository:
   `git clone https://github.com/MicheleMeazzini/ai-interior-designer.git`

2. Run the application:
   `uv run src/app.py`

3. Open the provided local URL (e.g., http://127.0.0.1:7860) in your web browser.

## 🛠️ Tech Stack
* **UI:** Gradio
* **Core Generation:** Stable Diffusion v1.5
* **Fine-Tuning:** LoRA 
* **Spatial Conditioning:** ControlNet (MLSD)
* **Environment:** Python & uv
