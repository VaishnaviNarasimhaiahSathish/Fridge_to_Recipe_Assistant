from pathlib import Path
import random

from PIL import Image, ImageDraw


DATASET_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/figures")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "valid", "test"]

GRID_ROWS = 4
GRID_COLS = 5
THUMBNAIL_SIZE = (220, 220)
RANDOM_SEED = 42


def get_image_files(image_dir: Path):
    if not image_dir.exists():
        return []

    return [
        file for file in image_dir.rglob("*")
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_grid(image_paths, output_path: Path, title: str):
    if not image_paths:
        print(f"No images found for {title}")
        return

    grid_width = GRID_COLS * THUMBNAIL_SIZE[0]
    grid_height = GRID_ROWS * THUMBNAIL_SIZE[1] + 40

    grid = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid)
    draw.text((10, 10), title, fill="black")

    for idx, image_path in enumerate(image_paths[: GRID_ROWS * GRID_COLS]):
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        x = col * THUMBNAIL_SIZE[0]
        y = row * THUMBNAIL_SIZE[1] + 40

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE)

            canvas = Image.new("RGB", THUMBNAIL_SIZE, color="white")
            paste_x = (THUMBNAIL_SIZE[0] - image.width) // 2
            paste_y = (THUMBNAIL_SIZE[1] - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))

            grid.paste(canvas, (x, y))

        except Exception as error:
            print(f"Could not load image {image_path}: {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved sample grid: {output_path}")


def main():
    random.seed(RANDOM_SEED)

    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        image_files = get_image_files(image_dir)

        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            continue

        sample_size = min(GRID_ROWS * GRID_COLS, len(image_files))
        sampled_images = random.sample(image_files, sample_size)

        output_path = OUTPUT_DIR / f"sample_grid_{split}.png"
        title = f"{split} split - {sample_size} random sample images"

        create_grid(sampled_images, output_path, title)


if __name__ == "__main__":
    main()