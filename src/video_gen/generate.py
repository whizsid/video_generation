from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from video_gen.config import (
    DEFAULT_NEGATIVE_PROMPT,
    FIRST_FRAME_FITS,
    GENERATION_RESOLUTIONS,
    GGUF_QUANTS,
    PROFILES,
    REF_MODES,
    UPSCALE_TARGETS,
    GenConfig,
)


def _profile_help() -> str:
    lines = []
    for name, p in PROFILES.items():
        w, h = GENERATION_RESOLUTIONS[p.resolution]
        frames = p.num_frames if p.num_frames is not None else "native for --seconds"
        lines.append(f"{name}: {p.description} ({w}x{h}, {frames} frames, {p.steps} steps)")
    return "auto picks t4 when CUDA is available, otherwise mac. " + "; ".join(lines)


def build_parser() -> argparse.ArgumentParser:
    defaults = GenConfig(prompt="")
    parser = argparse.ArgumentParser(
        prog="video-gen",
        description="Generate a 16:9 video from reference image(s) and a prompt with Wan2.1-VACE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", "-p", required=True, help="Text description of the video.")
    parser.add_argument(
        "--ref", "-r", dest="refs", action="append", type=Path, default=[],
        help="Reference image. Repeat for multiple references.",
    )
    parser.add_argument(
        "--ref-mode", choices=list(REF_MODES), default=defaults.ref_mode,
        help="subject: references show who/what appears in a newly generated scene. "
        "first-frame: the first --ref becomes frame 0 and is animated as-is (face, colors, "
        "environment, lighting kept); any further --ref images act as subject references.",
    )
    parser.add_argument(
        "--fit", choices=list(FIRST_FRAME_FITS), default=defaults.fit,
        help="first-frame mode, for photos that aren't 16:9: pad keeps the whole photo and outpaints "
        "the sides; crop center-crops it to fill the frame.",
    )
    parser.add_argument(
        "--strength", type=float, default=defaults.strength,
        help="first-frame mode: how far frames may depart from the photo. Lower starts every frame "
        "from the noised photo (smoother start, less motion); 1.0 starts from pure noise (more "
        "motion, but the scene may morph in the first frames).",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT,
                        help="What to avoid (ignored by the t4 profile, which runs without guidance).")
    parser.add_argument(
        "--out", "-o", type=Path, default=None,
        help="Output .mp4 path (default: outputs/clip_<timestamp>.mp4).",
    )
    parser.add_argument("--profile", choices=["auto", *PROFILES], default="auto", help=_profile_help())

    # Options below default to None so the chosen profile supplies the value.
    gen = parser.add_argument_group("generation (defaults come from --profile)")
    gen.add_argument(
        "--resolution", choices=list(GENERATION_RESOLUTIONS), default=None,
        help="16:9 size the model generates at ("
        + ", ".join(f"{k}={w}x{h}" for k, (w, h) in GENERATION_RESOLUTIONS.items())
        + "). Larger sizes may run out of memory.",
    )
    gen.add_argument("--frames", type=int, default=None,
                     help="Frames the model generates; must be 4k+1 (17, 33, 49, 65, 81).")
    gen.add_argument("--steps", type=int, default=None)
    gen.add_argument("--guidance", type=float, default=None,
                     help="Prompt adherence. Keep 1.0 with the distilled t4 model.")
    gen.add_argument(
        "--ref-strength", type=float, default=defaults.conditioning_scale,
        help="How strongly the references / first frame condition the video (VACE conditioning scale).",
    )
    gen.add_argument("--flow-shift", type=float, default=None)
    gen.add_argument("--fps", type=int, default=defaults.fps)
    gen.add_argument("--seed", type=int, default=None, help="Random if omitted.")

    post = parser.add_argument_group("post-processing")
    post.add_argument("--seconds", type=float, default=defaults.seconds,
                      help="Final duration: extra frames are trimmed, missing ones interpolated (macOS).")
    post.add_argument("--no-interpolate", action="store_true",
                      help="Keep exactly the generated frames (duration = frames / fps).")
    post.add_argument("--upscale", choices=list(UPSCALE_TARGETS), default=defaults.upscale,
                      help="Final 16:9 output size via Real-ESRGAN.")

    hw = parser.add_argument_group("hardware / memory (defaults come from --profile)")
    hw.add_argument("--quant", choices=GGUF_QUANTS, default=None,
                    help="GGUF quantization of the 14B model (t4 profile). Larger is better quality "
                         "but needs more VRAM and RAM.")
    hw.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    hw.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None,
                    help="Transformer dtype. On mac, try bfloat16 if float16 gives black frames.")
    hw.add_argument("--offload", choices=["sequential", "model", "none"], default=None,
                    help="sequential = lowest memory; none = fastest but needs the most memory.")
    hw.add_argument("--no-vae-tiling", action="store_true", help="Disable tiled VAE decoding.")
    hw.add_argument("--conv3d-patch", action="store_true",
                    help="Use the native Metal Conv3D kernel on mac (requires: uv sync --extra conv3d).")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def resolve_profile(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "t4" if torch.cuda.is_available() else "mac"


def config_from_args(args: argparse.Namespace) -> GenConfig:
    out = args.out or Path("outputs") / f"clip_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    return GenConfig.from_profile(
        resolve_profile(args.profile),
        prompt=args.prompt,
        reference_paths=args.refs,
        ref_mode=args.ref_mode,
        fit=args.fit,
        strength=args.strength,
        negative_prompt=args.negative_prompt,
        output_path=out,
        resolution=args.resolution,
        num_frames=args.frames,
        seconds=None if args.no_interpolate else args.seconds,
        upscale=args.upscale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        conditioning_scale=args.ref_strength,
        flow_shift=args.flow_shift,
        fps=args.fps,
        seed=args.seed,
        quant=args.quant,
        device=args.device,
        dtype=args.dtype,
        offload=args.offload,
        vae_tiling=not args.no_vae_tiling,
        use_conv3d_patch=args.conv3d_patch,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config_from_args(args)
    try:
        cfg.validate()
    except (ValueError, FileNotFoundError) as exc:
        logging.error("%s", exc)
        return 2
    logging.info(
        "Profile %s: %s, %dx%d, %d frames, %d steps%s, ref-mode %s",
        cfg.profile, cfg.model_id, cfg.width, cfg.height, cfg.num_frames, cfg.num_inference_steps,
        f", GGUF {cfg.quant}" if cfg.gguf_repo else "", cfg.ref_mode,
    )

    # Imported lazily so `--help` and argument errors don't pay the torch/diffusers import cost.
    from video_gen.pipeline import generate

    generate(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
