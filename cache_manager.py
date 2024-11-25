import hashlib
import pickle
from pathlib import Path
from typing import Optional
from PIL import Image
import os

CACHE_DIR = Path(".cache")
CACHE_SIZE_LIMIT = 100  # Maximum number of cached images

def get_cache_key(prompt: str, size: int, pixel_size: int, style_strength: float) -> str:
    """Generate a unique cache key for the given parameters"""
    params = f"{prompt}{size}{pixel_size}{style_strength}"
    return hashlib.md5(params.encode()).hexdigest()

def get_cached_result(
    prompt: str,
    size: int,
    pixel_size: int,
    style_strength: float
) -> Optional[Image.Image]:
    """Retrieve cached result if available"""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_key = get_cache_key(prompt, size, pixel_size, style_strength)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

def cache_result(
    prompt: str,
    image: Image.Image,
    size: int,
    pixel_size: int,
    style_strength: float
) -> None:
    """Cache the generated image"""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_key = get_cache_key(prompt, size, pixel_size, style_strength)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        # Clean up old cache files if limit is reached
        cleanup_cache()
        
        with open(cache_file, 'wb') as f:
            pickle.dump(image, f)
    except Exception:
        pass

def cleanup_cache():
    """Remove old cache files if cache size limit is reached"""
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        if len(cache_files) > CACHE_SIZE_LIMIT:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            for file in cache_files[:-CACHE_SIZE_LIMIT]:
                file.unlink()
    except Exception:
        pass
