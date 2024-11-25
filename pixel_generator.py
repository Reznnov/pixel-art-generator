import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

class PixelArtGenerator:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-2-1"
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
        # Enhance prompt for pixel art style
        enhanced_prompt = f"pixel art style, 8-bit, {prompt}, high quality, detailed"
        negative_prompt = "blur, realistic, 3d, photographic, high resolution"
        
        # Generate image
        image = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=size,
            height=size
        ).images[0]
        
        return image

    def __del__(self):
        # Clean up CUDA memory
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
