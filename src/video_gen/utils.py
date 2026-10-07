from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_reference_image(path: Path, height: int, width: int) -> Image.Image:
    """Load an image as RGB and downscale it to fit inside (height, width).

    The pipeline letterboxes each reference onto a white canvas of the target size,
    so we only shrink oversized images here to avoid huge intermediate tensors.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")

        if img.height > height or img.width > width:
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
        return img.copy()


def load_reference_images(paths: list[Path], height: int, width: int) -> list[Image.Image]:
    return [load_reference_image(p, height, width) for p in paths]


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
