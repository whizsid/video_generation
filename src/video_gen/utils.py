from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


# VACE convention: mid-gray pixels are "unknown"; in masks, white = generate and black = keep.
UNKNOWN_PIXEL = (128, 128, 128)
MASK_GENERATE = 255
MASK_KEEP = 0


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return img.convert("RGB")


def load_reference_image(path: Path, height: int, width: int) -> Image.Image:
    """Load an image as RGB and downscale it to fit inside (height, width).

    The pipeline letterboxes each reference onto a white canvas of the target size,
    so we only shrink oversized images here to avoid huge intermediate tensors.
    """
    img = _load_rgb(path)
    if img.height > height or img.width > width:
        img.thumbnail((width, height), Image.Resampling.LANCZOS)
    return img


def load_reference_images(paths: list[Path], height: int, width: int) -> list[Image.Image]:
    return [load_reference_image(p, height, width) for p in paths]


def build_first_frame_conditioning(
    path: Path, height: int, width: int, num_frames: int, fit: str
) -> tuple[list[Image.Image], list[Image.Image]]:
    """VACE video + mask that keep the photo as frame 0 and generate every later frame.

    fit="crop" fills the frame by center-cropping the photo; fit="pad" keeps the whole photo
    centered and lets the model outpaint the bars around it.
    """
    photo = _load_rgb(path)
    if fit == "crop":
        first = ImageOps.fit(photo, (width, height), Image.Resampling.LANCZOS)
        first_mask = Image.new("L", (width, height), MASK_KEEP)
    else:
        photo = ImageOps.contain(photo, (width, height), Image.Resampling.LANCZOS)
        left, top = (width - photo.width) // 2, (height - photo.height) // 2
        first = Image.new("RGB", (width, height), UNKNOWN_PIXEL)
        first.paste(photo, (left, top))
        first_mask = Image.new("L", (width, height), MASK_GENERATE)
        first_mask.paste(MASK_KEEP, (left, top, left + photo.width, top + photo.height))

    unknown = Image.new("RGB", (width, height), UNKNOWN_PIXEL)
    generate = Image.new("L", (width, height), MASK_GENERATE)
    return [first] + [unknown] * (num_frames - 1), [first_mask] + [generate] * (num_frames - 1)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
