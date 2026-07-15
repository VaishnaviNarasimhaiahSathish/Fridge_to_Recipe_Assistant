"""Resolve fridge image files across curated and raw dataset folders."""

from pathlib import Path


IMAGE_LOOKUP_DIRS = (
    Path("data/annotations/final_images_200"),
    Path("data/raw/train/images"),
    Path("data/raw/valid/images"),
    Path("data/raw/test/images"),
)


def resolve_image_path(stored_path, project_root: Path | None = None) -> Path | None:
    """Find an image file by filename.

    The recorded image_path often points to the raw train/valid/test split,
    but curated evaluation/demo images may live in:

        data/annotations/final_images_200

    Only the filename is trusted. The curated folder is checked first because
    it is the final image set used by the UI and 100-image evaluation.

    Returns the resolved Path, or None if the file cannot be found.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    filename = Path(stored_path).name

    for image_dir in IMAGE_LOOKUP_DIRS:
        candidate = root / image_dir / filename

        if candidate.exists():
            return candidate

    return None