from PIL import Image
import numpy as np

def post_process_image(image: Image.Image, pixel_size: int = 4) -> Image.Image:
    """
    Apply pixel art post-processing effects with enhanced quality
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Increase contrast
    mean = np.mean(img_array)
    img_array = np.clip((img_array - mean) * 1.2 + mean, 0, 255).astype(np.uint8)
    
    # Convert back to PIL and reduce colors using optimized quantization
    img = Image.fromarray(img_array)
    img = img.quantize(colors=32, method=2, dither=Image.Dither.NONE)
    img = img.convert('RGB')
    
    # Calculate new dimensions ensuring even division
    width, height = img.size
    new_width = (width // pixel_size) * pixel_size
    new_height = (height // pixel_size) * pixel_size
    
    # Crop to ensure clean pixel boundaries
    img = img.crop((0, 0, new_width, new_height))
    
    # Optimize pixelization
    small_img = img.resize((new_width // pixel_size, new_height // pixel_size), Image.Resampling.LANCZOS)
    img = small_img.resize((new_width, new_height), Image.Resampling.NEAREST)
    
    return img
