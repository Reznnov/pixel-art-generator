import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from PIL import Image
import numpy as np
import streamlit as st
import gc
import os

@st.cache_resource
def get_pipeline():
    """Initialize and return the StableDiffusion pipeline with optimized settings"""
    with st.spinner("Загрузка AI модели..."):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "CompVis/stable-diffusion-v1-4"
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
            ).to(device)
            
            # Optimize memory usage
            pipe.enable_attention_slicing()
            if device == "cuda":
                pipe.enable_vae_tiling()
            
            return pipe
            
        except Exception as e:
            st.error(f"Ошибка при инициализации модели: {str(e)}")
            raise

class PixelArtGenerator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                cls._instance.pipe = get_pipeline()
            except Exception as e:
                st.error("Не удалось инициализировать генератор. Попробуйте перезагрузить страницу.")
                raise
        return cls._instance

    def generate(self, prompt: str, size: int = 128, style_strength: float = 0.8) -> Image.Image:
        """
        Generate high-quality pixel art from a text prompt
        """
        # Enhance prompt for pixel art style with improved aesthetics
        enhanced_prompt = f"pixel art style, {prompt}, highly detailed pixel art, 8-bit style, vibrant colors, clear outlines, clean pixel art, {prompt}, sharp pixels, retro game art style, clear composition"
        negative_prompt = "blur, realistic, 3d, photographic, high resolution, painting, anime, sketch, watercolor, abstract, distorted, blurry, noisy, messy, undefined"
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
            progress = min((step + 1) / 30, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Генерация изображения... Шаг {step + 1}/30")
            
        try:
            # Generate image with optimizations
            with torch.no_grad():
                image = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=12.0,
                    width=size,
                    height=size,
                    callback=callback_fn,
                    callback_steps=1,
                    batch_size=1  # Оптимизация использования памяти
                ).images[0]
                
            # Clear CUDA cache after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()  # Запуск сборщика мусора
                
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return image
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ошибка при генерации изображения: {str(e)}")
            raise
            
    def __del__(self):
        # Clean up CUDA memory
        if hasattr(self, 'pipe'):
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
