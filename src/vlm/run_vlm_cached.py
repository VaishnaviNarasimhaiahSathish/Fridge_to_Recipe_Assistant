import base64
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from vlm_prediction_cache import (
    get_cached_prediction,
    load_cache,
    save_cache,
    store_prediction,
    cache_stats,
)


BASE_URL    = "https://llms.innkube.fim.uni-passau.de"
PROMPT_PATH = Path("configs/vlm_prompt_with_counts.txt")
CACHE_PATH  = Path("reports/vlm_prediction_cache.json")

# Resize images before sending to reduce inference time
MAX_IMAGE_SIZE = 512


def load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_image_as_base64(image_path: Path) -> tuple[str, str]:
    """
    Encode image as base64. Resizes to MAX_IMAGE_SIZE on longest side
    before encoding to reduce VLM inference time.
    """
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")

    # Resize — keeps aspect ratio, reduces token count
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded, "image/jpeg"


def query_vlm(image_path: Path, client: OpenAI, prompt: str) -> dict:
    image_base64, mime_type = encode_image_as_base64(image_path)

    start = time.time()

    response = client.chat.completions.create(
        model="qwen35-397b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    elapsed = time.time() - start
    raw_response = response.choices[0].message.content

    # Parse JSON from response
    parsed = None
    try:
        import re
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    return {
        "image_id":        image_path.name,
        "image_path":      str(image_path),
        "status":          "success" if parsed else "parse_error",
        "elapsed_seconds": round(elapsed, 2),
        "raw_response":    raw_response,
        "parsed_response": parsed,
    }


def run_with_cache(image_paths: list[Path], force_refresh: bool = False) -> list[dict]:
    """
    Query VLM for each image, using cache where available.
    Skips API call if prediction already cached and force_refresh=False.
    """
    api_key = os.getenv("INNKUBE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "INNKUBE_API_KEY is not set. Run:\n"
            "set INNKUBE_API_KEY=your_key_here"
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    prompt = load_prompt(PROMPT_PATH)
    cache  = load_cache(CACHE_PATH)

    stats = cache_stats(cache)
    print(f"Cache loaded: {stats['total_cached']} existing predictions")
    print()

    results     = []
    cache_hits  = 0
    api_calls   = 0

    for i, image_path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] {image_path.name} ... ", end="", flush=True)

        if not force_refresh:
            cached = get_cached_prediction(image_path, cache)
            if cached:
                print("cache hit")
                results.append(cached)
                cache_hits += 1
                continue

        # Not cached — query the API
        result = query_vlm(image_path, client, prompt)
        cache  = store_prediction(image_path, result, cache)
        save_cache(cache, CACHE_PATH)

        status = result["status"]
        elapsed = result["elapsed_seconds"]
        print(f"{status} ({elapsed:.1f}s)")

        results.append(result)
        api_calls += 1

    print()
    print(f"Done. Cache hits: {cache_hits}  API calls: {api_calls}")
    print(f"Cache saved to: {CACHE_PATH}")

    return results


def main():
    # Example: run on a small sample of images from the dataset
    image_dir = Path("data/raw/valid/images")

    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        print("Update image_dir to point to your local dataset images.")
        return

    image_paths = sorted(image_dir.glob("*.jpg"))[:5]

    if not image_paths:
        print("No images found.")
        return

    print(f"Running cached VLM on {len(image_paths)} images")
    print(f"Max image size: {MAX_IMAGE_SIZE}px (resized before sending)")
    print()

    results = run_with_cache(image_paths)

    print()
    print("Sample result:")
    r = results[0]
    print(f"  Image   : {r['image_id']}")
    print(f"  Status  : {r['status']}")
    print(f"  Elapsed : {r.get('elapsed_seconds', 'cached')}s")
    if r.get("parsed_response"):
        ings = r["parsed_response"].get("ingredients", [])
        print(f"  Ingredients: {[i['name'] for i in ings]}")


if __name__ == "__main__":
    main()