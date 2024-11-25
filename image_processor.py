from PIL import Image
import numpy as np

def post_process_image(image: Image.Image, pixel_size: int = 4) -> Image.Image:
    """
    Apply pixel art post-processing effects
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Reduce colors using quantization
    img = Image.fromarray(img_array)
    img = img.quantize(colors=256, method=2)
    img = img.convert('RGB')
    
    # Calculate new dimensions
    width, height = img.size
    new_width = width // pixel_size
    new_height = height // pixel_size
    
    # Resize down and up to create pixel effect
    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    img = img.resize((width, height), Image.Resampling.NEAREST)
    
    return img
