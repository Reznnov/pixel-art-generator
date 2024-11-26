import os
from huggingface_hub import InferenceClient
from PIL import Image
import streamlit as st

@st.cache_resource
def get_client():
    with st.spinner("Загрузка AI модели..."):
        try:
            return InferenceClient(
                "nerijs/pixel-art-xl",
                token=os.environ["HUGGINGFACE_TOKEN"]
            )
        except Exception as e:
            st.error(f"Ошибка при инициализации модели: {str(e)}")
            raise

class PixelArtGenerator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance.client = get_client()
            except Exception as e:
                st.error("Не удалось инициализировать генератор. Попробуйте перезагрузить страницу.")
                raise
        return cls._instance

    def generate(self, prompt: str, size: int = 128, style_strength: float = 0.8) -> Image.Image:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Генерация пиксель арта...")
            progress_bar.progress(0.5)
            
            image = self.client.text_to_image(prompt)
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            return image
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ошибка при генерации изображения: {str(e)}")
            raise