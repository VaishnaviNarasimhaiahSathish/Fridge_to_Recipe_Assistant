import hashlib
import json
from pathlib import Path


CACHE_PATH = Path("reports/vlm_prediction_cache.json")


def _image_hash(image_path: Path) -> str:
    """SHA-256 hash of image file bytes — used as cache key."""
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_cache(cache_path: Path = CACHE_PATH) -> dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_path: Path = CACHE_PATH):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_cached_prediction(image_path: Path, cache: dict) -> dict | None:
    """
    Return cached VLM prediction for this image if it exists.
    Returns None if not cached.
    """
    key = _image_hash(image_path)
    return cache.get(key)


def store_prediction(image_path: Path, prediction: dict, cache: dict) -> dict:
    """
    Store a VLM prediction in the cache dict and return the updated cache.
    Call save_cache() afterwards to persist to disk.
    """
    key = _image_hash(image_path)
    cache[key] = prediction
    return cache


def cache_stats(cache: dict) -> dict:
    return {
        "total_cached": len(cache),
    }