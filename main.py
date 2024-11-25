import streamlit as st
from pixel_generator import PixelArtGenerator
from image_processor import post_process_image
from cache_manager import get_cached_result, cache_result
from utils import validate_prompt, setup_page
import io

def main():
    setup_page()
    
    st.title("🎨 Pixel Art Generator")
    st.write("Generate unique pixel art from text descriptions!")

    # Sidebar controls
    st.sidebar.header("Generation Settings")
    
    image_size = st.sidebar.select_slider(
        "Image Size",
        options=[32, 64, 128],
        value=64
    )
    
    pixel_size = st.sidebar.slider(
        "Pixel Size",
        min_value=2,
        max_value=8,
        value=4
    )
    
    style_strength = st.sidebar.slider(
        "Pixelation Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1
    )

    # Main interface
    prompt = st.text_area(
        "Enter your description",
        placeholder="Example: a cute pixel art cat in space"
    )

    if st.button("Generate Art", type="primary"):
        if not validate_prompt(prompt):
            st.error("Please enter a valid prompt (minimum 3 characters)")
            return

        try:
            import signal
            from contextlib import contextmanager
            import torch
            
            @contextmanager
            def timeout(seconds):
                def handler(signum, frame):
                    raise TimeoutError("Generation took too long!")
                
                # Register the signal function handler
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                
                try:
                    yield
                finally:
                    # Disable the alarm
                    signal.alarm(0)
            
            # Check cache first
            st.info("Checking cache for similar generations...")
            cached_result = get_cached_result(prompt, image_size, pixel_size, style_strength)
            
            if cached_result is not None:
                st.success("Found cached result!")
                generated_image = cached_result
            else:
                st.info("No cached version found. Starting new generation...")
                
                try:
                    with timeout(180):  # 3 minutes timeout
                        # Generate new image
                        generator = PixelArtGenerator()
                        raw_image = generator.generate(
                            prompt,
                            size=image_size,
                            style_strength=style_strength
                        )
                        
                        st.info("Applying pixel art post-processing...")
                        # Post-process the image
                        generated_image = post_process_image(
                            raw_image,
                            pixel_size=pixel_size
                        )
                        
                        st.info("Caching the result...")
                        # Cache the result
                        cache_result(prompt, generated_image, image_size, pixel_size, style_strength)
                        
                except torch.cuda.OutOfMemoryError:
                    st.error("Не хватает памяти GPU. Попробуйте уменьшить размер изображения или очистить кэш GPU.")
                    return
                except TimeoutError:
                    st.error("Генерация заняла слишком много времени. Попробуйте еще раз или измените параметры.")
                    return

            # Display the result
            st.success("Generation completed successfully!")
            st.image(generated_image, caption="Generated Pixel Art")
                
                # Add download button
            buf = io.BytesIO()
            generated_image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="pixel_art.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
