import streamlit as st
from pixel_generator import PixelArtGenerator
from image_processor import post_process_image
from cache_manager import get_cached_result, cache_result
from utils import validate_prompt, setup_page
import io
from threading import Timer

def timeout_handler():
    st.session_state.generation_timeout = True

def main():
    if 'generation_timeout' not in st.session_state:
        st.session_state.generation_timeout = False
    setup_page()
    
    st.title("🎨 Pixel Art Generator")
    st.write("Generate unique pixel art from text descriptions!")
    
    st.markdown('''
    💡 **Tips for better results:**
    - Be specific in your description
    - Include colors and details
    - Specify the perspective (front view, side view, etc.)
    - Example: "a cute orange cat wearing a space helmet, front view, pixel art style"
    ''')

    # Sidebar controls
    st.sidebar.header("Generation Settings")
    
    image_size = st.sidebar.select_slider(
        "Image Size",
        options=[32, 64, 128],
        value=128
    )
    
    pixel_size = st.sidebar.slider(
        "Pixel Size",
        min_value=2,
        max_value=8,
        value=3
    )
    
    style_strength = st.sidebar.slider(
        "Pixelation Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
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
            # Check cache first
            st.info("Checking cache for similar generations...")
            cached_result = get_cached_result(prompt, image_size, pixel_size, style_strength)
            
            if cached_result is not None:
                st.success("Found cached result!")
                generated_image = cached_result
            else:
                st.info("No cached version found. Starting new generation...")
                
                # Set up timeout
                timer = Timer(300.0, timeout_handler)  # 5 minutes timeout
                max_retries = 3
                retry_count = 0
                network_error = None

                try:
                    timer.start()
                    # Generate new image with retries
                    while retry_count < max_retries:
                        try:
                            generator = PixelArtGenerator()
                            raw_image = generator.generate(
                                prompt,
                                size=image_size,
                                style_strength=style_strength
                            )
                            network_error = None
                            break
                        except (ConnectionError, TimeoutError) as e:
                            network_error = e
                            retry_count += 1
                            if retry_count < max_retries:
                                st.warning(f"Попытка подключения {retry_count} из {max_retries}...")
                                continue
                            raise ConnectionError("Не удалось подключиться к серверу после нескольких попыток. Пожалуйста, проверьте подключение к интернету.")

                    if st.session_state.generation_timeout:
                        raise TimeoutError("Generation took too long!")
                    
                    if network_error:
                        raise network_error
                    
                    st.info("Applying pixel art post-processing...")
                    # Validate and post-process the image
                    if raw_image is not None:
                        generated_image = post_process_image(raw_image, pixel_size=pixel_size)
                    else:
                        raise ValueError("Не удалось сгенерировать изображение")
                    
                    st.info("Caching the result...")
                    # Cache the result
                    cache_result(prompt, generated_image, image_size, pixel_size, style_strength)
                    
                except TimeoutError:
                    st.error("Генерация заняла слишком много времени. Попробуйте еще раз или измените параметры.")
                    return
                finally:
                    timer.cancel()
                    st.session_state.generation_timeout = False

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

        except ConnectionError as e:
            st.error(f"Ошибка сети: {str(e)}\nПожалуйста, проверьте подключение к интернету и попробуйте снова.")
        except TimeoutError as e:
            st.error(f"Превышено время ожидания: {str(e)}\nСервер не отвечает, попробуйте позже.")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}\nПожалуйста, попробуйте еще раз или обратитесь в поддержку.")

if __name__ == "__main__":
    main()
