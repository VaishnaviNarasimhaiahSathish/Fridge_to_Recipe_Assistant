"""Resolve fridge image files across the train/valid/test dataset splits."""

from pathlib import Path

IMAGE_SPLIT_DIRS = (
    Path("data/raw/train/images"),
    Path("data/raw/valid/images"),
    Path("data/raw/test/images"),
)


def resolve_image_path(stored_path, project_root: Path | None = None) -> Path | None:
    """Find an image's file by trying each dataset split in turn.

    The recorded ``image_path`` often bakes in the split it was annotated
    from (e.g. ``data/raw/valid/images/foo.jpg``), but images can move
    between splits over time. Only the filename is trusted; train, valid,
    and test are tried in that order, and the first match wins.

    Returns the resolved Path, or None if the file isn't in any split.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    filename = Path(stored_path).name

    for split_dir in IMAGE_SPLIT_DIRS:
        candidate = root / split_dir / filename
        if candidate.exists():
            return candidate

    return None
