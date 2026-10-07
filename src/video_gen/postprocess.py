from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from video_gen.config import UPSCALE_TARGETS, GenConfig
from video_gen.utils import format_duration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    name: str
    url: str
    download_mb: int
    binary: str
    # Archive members to extract (prefix match); everything else in the zip is skipped.
    members: tuple[str, ...]
    model_dir: str


REALESRGAN = Tool(
    name="realesrgan",
    url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip",
    download_mb=52,
    binary="realesrgan-ncnn-vulkan",
    members=("realesrgan-ncnn-vulkan", "models/realesrgan-x4plus."),
    model_dir="models",
)
RIFE = Tool(
    name="rife",
    url="https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip",
    download_mb=437,
    binary="rife-ncnn-vulkan-20221029-macos/rife-ncnn-vulkan",
    members=("rife-ncnn-vulkan-20221029-macos/rife-ncnn-vulkan", "rife-ncnn-vulkan-20221029-macos/rife-v4.6/"),
    model_dir="rife-ncnn-vulkan-20221029-macos/rife-v4.6",
)

# realesrgan-x4plus is trained on real-world photos and only supports 4x.
UPSCALE_MODEL = "realesrgan-x4plus"
UPSCALE_FACTOR = 4
# PyTorch weights of the same model, used through spandrel where the macOS ncnn build can't run.
REALESRGAN_TORCH_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def ensure_tool(tool: Tool, tools_dir: Path) -> Path:
    """Download and unpack a tool's macOS build on first use; returns the tool's root directory."""
    root = tools_dir / tool.name
    if (root / tool.binary).is_file():
        return root

    root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (~%d MB, one time only)...", tool.name, tool.download_mb)
    with tempfile.NamedTemporaryFile(dir=tools_dir, suffix=".zip") as archive:
        with urllib.request.urlopen(tool.url) as response:
            shutil.copyfileobj(response, archive)
        archive.flush()
        with zipfile.ZipFile(archive.name) as zf:
            for member in zf.namelist():
                if member.startswith(tool.members) and not member.endswith("/"):
                    zf.extract(member, root)

    binary = root / tool.binary
    binary.chmod(0o755)
    return root


def _run(cmd: list[str | Path]) -> None:
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed (exit {result.returncode}):\n{tail}")


def _write_frames(frames: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(_to_uint8(frame)).save(out_dir / f"{i:08d}.png")


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return (np.clip(frame, 0.0, 1.0) * 255).round().astype(np.uint8)


def interpolate(in_dir: Path, out_dir: Path, target_count: int, tools_dir: Path) -> None:
    root = ensure_tool(RIFE, tools_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([root / RIFE.binary, "-i", in_dir, "-o", out_dir, "-n", target_count, "-m", root / RIFE.model_dir])


def upscale(in_dir: Path, out_dir: Path, tools_dir: Path) -> None:
    root = ensure_tool(REALESRGAN, tools_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([
        root / REALESRGAN.binary, "-i", in_dir, "-o", out_dir,
        "-s", UPSCALE_FACTOR, "-n", UPSCALE_MODEL, "-m", root / REALESRGAN.model_dir,
    ])


def _ensure_file(url: str, dest: Path) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (one time only)...", dest.name)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.rename(dest)
    return dest


def upscale_torch(in_dir: Path, out_dir: Path, tools_dir: Path, device: str) -> None:
    """Real-ESRGAN x4plus in PyTorch (via spandrel), for CUDA machines such as Colab."""
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    weights = _ensure_file(REALESRGAN_TORCH_URL, tools_dir / "realesrgan-torch" / "RealESRGAN_x4plus.pth")
    model = ModelLoader().load_from_file(weights)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(f"{weights} is not an image-to-image model.")
    model.to(device).eval()
    if device == "cuda" and model.supports_half:
        model.half()

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for frame_path in sorted(in_dir.glob("*.png")):
            with Image.open(frame_path) as img:
                pixels = np.asarray(img.convert("RGB"))
            tensor = torch.from_numpy(pixels).to(device).permute(2, 0, 1).unsqueeze(0)
            output = model(tensor.to(model.dtype).div_(255))
            result = output[0].permute(1, 2, 0).float().mul_(255).round_().clamp_(0, 255)
            Image.fromarray(result.byte().cpu().numpy()).save(out_dir / frame_path.name)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def write_video(frame_paths: list[Path], path: Path, fps: int, size: tuple[int, int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=8
    ) as writer:
        for frame_path in frame_paths:
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                if size and img.size != size:
                    img = img.resize(size, Image.Resampling.LANCZOS)
                writer.append_data(np.asarray(img))


def needs_postprocess(cfg: GenConfig) -> bool:
    target = UPSCALE_TARGETS[cfg.upscale]
    return cfg.target_frame_count != cfg.num_frames or (target is not None and target[0] > cfg.width)


def postprocess(frames: np.ndarray, cfg: GenConfig) -> None:
    """Trim or interpolate to the target duration at the generated size, then upscale and encode."""
    target_size = UPSCALE_TARGETS[cfg.upscale]
    target_frames = cfg.target_frame_count
    do_upscale = target_size is not None and target_size[0] > frames.shape[2]
    on_macos = sys.platform == "darwin"

    if len(frames) > target_frames:
        logger.info("Trimming %d -> %d frames (%.1fs at %d fps)", len(frames), target_frames,
                    target_frames / cfg.fps, cfg.fps)
        frames = frames[:target_frames]

    with tempfile.TemporaryDirectory(prefix="video_gen_") as tmp:
        current = Path(tmp) / "frames"
        _write_frames(frames, current)

        if target_frames > len(frames):
            if on_macos:
                start = time.perf_counter()
                logger.info("Interpolating %d -> %d frames with RIFE...", len(frames), target_frames)
                interpolate(current, Path(tmp) / "interpolated", target_frames, cfg.tools_dir)
                current = Path(tmp) / "interpolated"
                logger.info("Interpolation took %s", format_duration(time.perf_counter() - start))
            else:
                logger.warning(
                    "Frame interpolation uses the macOS RIFE build; keeping the %d generated frames "
                    "(%.1fs). Increase --frames to generate a longer clip natively.",
                    len(frames), len(frames) / cfg.fps,
                )

        if do_upscale:
            start = time.perf_counter()
            logger.info("Upscaling to %dx%d with Real-ESRGAN (%s)...", *target_size, UPSCALE_MODEL)
            if on_macos:
                upscale(current, Path(tmp) / "upscaled", cfg.tools_dir)
            else:
                upscale_torch(current, Path(tmp) / "upscaled", cfg.tools_dir, cfg.device)
            current = Path(tmp) / "upscaled"
            logger.info("Upscaling took %s", format_duration(time.perf_counter() - start))

        frame_paths = sorted(current.glob("*.png"))
        write_video(frame_paths, cfg.output_path, cfg.fps, target_size if do_upscale else None)
        logger.info(
            "Final video: %d frames, %.1fs at %d fps", len(frame_paths), len(frame_paths) / cfg.fps, cfg.fps
        )
