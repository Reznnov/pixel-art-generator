import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from PIL import Image
import numpy as np

class PixelArtGenerator:
    def __init__(self):
        self.model_id = "CompVis/stable-diffusion-v1-4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Optimize memory usage
        self.pipe.enable_attention_slicing()
        if self.device == "cuda":
            self.pipe.enable_vae_tiling()

    def generate(self, prompt: str, size: int = 64, style_strength: float = 0.8) -> Image.Image:
        """
        Generate pixel art from a text prompt
        """
        import streamlit as st

        # Enhance prompt for pixel art style
        enhanced_prompt = f"pixel art style, {prompt}, highly detailed pixel art, 16-bit, clean pixel art, {prompt}, sharp pixels, retro game art style, clear composition"
        negative_prompt = "blur, realistic, 3d, photographic, high resolution, painting, anime, sketch, watercolor, abstract, distorted"
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
            progress = (step + 1) / 30  # 30 is the new num_inference_steps
            progress_bar.progress(progress)
            status_text.text(f"Generating image... Step {step + 1}/30")
            
        # Generate image with optimizations
        with torch.no_grad():
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=9.0,
                width=size,
                height=size,
                callback=callback_fn,
                callback_steps=1
            ).images[0]
            
        # Clear CUDA cache after generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return image

    def __del__(self):
        # Clean up CUDA memory
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
