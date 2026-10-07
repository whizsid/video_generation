from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

MODEL_ID_1_3B = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
# The 14B repo supplies the config, text encoder, tokenizer and VAE; the transformer comes from GGUF.
MODEL_ID_14B = "Wan-AI/Wan2.1-VACE-14B-diffusers"
# VACE-14B with the LightX2V step + CFG distillation merged in: 4-8 steps, guidance 1.0.
GGUF_REPO_14B = "QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF"
GGUF_FILE_TEMPLATE = "Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-{quant}.gguf"
GGUF_QUANTS = (
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

Device = Literal["mps", "cuda", "cpu"]
OffloadMode = Literal["sequential", "model", "none"]
DType = Literal["float16", "bfloat16", "float32"]
SchedulerName = Literal["unipc", "euler"]

# Height/width must be divisible by VAE spatial factor (8) x transformer patch size (2).
SPATIAL_MULTIPLE = 16
# The VAE compresses time by 4, so frame counts must be 4k + 1.
TEMPORAL_MULTIPLE = 4
# Wan2.1 is trained on clips of up to 81 frames (~5 s at 16 fps).
MAX_FRAMES = 81

# Exact 16:9 sizes whose sides are also multiples of 16 (width = 256 * m, height = 144 * m).
# 256x144 is excluded: the model produces incoherent color noise at that size.
GENERATION_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "288p": (512, 288),
    "432p": (768, 432),
    "576p": (1024, 576),
    "720p": (1280, 720),
}
UPSCALE_TARGETS: dict[str, tuple[int, int] | None] = {
    "none": None,
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


@dataclass(frozen=True)
class Profile:
    description: str
    model_id: str
    device: Device
    dtype: DType
    offload: OffloadMode
    text_encoder_device: Device
    text_encoder_dtype: DType
    resolution: str
    # None means: generate every frame needed for `seconds` natively instead of interpolating.
    num_frames: int | None
    steps: int
    guidance: float
    flow_shift: float
    scheduler: SchedulerName
    gguf_repo: str | None = None
    quant: str | None = None


PROFILES: dict[str, Profile] = {
    "mac": Profile(
        description="Apple Silicon, 8 GB+: Wan2.1-VACE-1.3B on MPS",
        model_id=MODEL_ID_1_3B,
        device="mps",
        # MPS on M1 has no fast bfloat16; bfloat16 is a fallback if float16 yields NaNs.
        dtype="float16",
        offload="sequential",
        # The 11 GB text encoder can't share 8 GB with the transformer, so it runs on CPU first.
        text_encoder_device="cpu",
        text_encoder_dtype="bfloat16",
        resolution="288p",
        num_frames=33,
        steps=20,
        guidance=5.0,
        flow_shift=3.0,
        scheduler="unipc",
    ),
    "t4": Profile(
        description="NVIDIA T4 16 GB (Colab): Wan2.1-VACE-14B + LightX2V, GGUF",
        model_id=MODEL_ID_14B,
        device="cuda",
        # Turing GPUs have no bfloat16 support.
        dtype="float16",
        # Free Colab has ~12.7 GB of RAM, too little to offload a 14B model to the CPU.
        offload="none",
        text_encoder_device="cuda",
        text_encoder_dtype="float16",
        # 768x432 x 65 frames is the largest size whose activations fit next to Q3_K_M in 15 GB.
        resolution="432p",
        num_frames=None,
        steps=6,
        guidance=1.0,
        flow_shift=5.0,
        scheduler="euler",
        gguf_repo=GGUF_REPO_14B,
        # Q3_K_M (8.6 GB) is the largest quant that leaves room for activations in T4 VRAM.
        quant="Q3_K_M",
    ),
}


def frames_for_seconds(seconds: float, fps: int) -> int:
    """Smallest valid (4k+1) frame count covering `seconds`, capped at the model's maximum."""
    needed = max(1, math.ceil(seconds * fps))
    k = math.ceil((needed - 1) / TEMPORAL_MULTIPLE)
    return min(MAX_FRAMES, TEMPORAL_MULTIPLE * k + 1)


@dataclass
class GenConfig:
    prompt: str
    reference_paths: list[Path] = field(default_factory=list)
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_path: Path = Path("outputs/clip.mp4")

    profile: str = "mac"
    width: int = 512
    height: int = 288
    num_frames: int = 33
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    conditioning_scale: float = 1.0
    flow_shift: float = 3.0
    scheduler: SchedulerName = "unipc"
    fps: int = 16
    seed: int | None = None
    max_sequence_length: int = 512

    # Post-processing: trim or RIFE-interpolate to `seconds`, then Real-ESRGAN upscales to `upscale`.
    seconds: float | None = 4.0
    upscale: str = "720p"
    tools_dir: Path = Path(".tools")

    model_id: str = MODEL_ID_1_3B
    gguf_repo: str | None = None
    quant: str | None = None
    device: Device = "mps"
    dtype: DType = "float16"
    offload: OffloadMode = "sequential"
    text_encoder_device: Device = "cpu"
    text_encoder_dtype: DType = "bfloat16"
    vae_tiling: bool = True
    use_conv3d_patch: bool = False
    embeds_cache_dir: Path = Path(".cache/prompt_embeds")

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> GenConfig:
        """Build a config from a profile's defaults; `None` overrides keep the profile value."""
        p = PROFILES[profile]
        width, height = GENERATION_RESOLUTIONS[overrides.pop("resolution", None) or p.resolution]
        values: dict[str, Any] = {
            "profile": profile,
            "model_id": p.model_id,
            "gguf_repo": p.gguf_repo,
            "quant": p.quant,
            "device": p.device,
            "dtype": p.dtype,
            "offload": p.offload,
            "text_encoder_device": p.text_encoder_device,
            "text_encoder_dtype": p.text_encoder_dtype,
            "width": width,
            "height": height,
            "num_inference_steps": p.steps,
            "guidance_scale": p.guidance,
            "flow_shift": p.flow_shift,
            "scheduler": p.scheduler,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"Unknown config fields: {', '.join(sorted(unknown))}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        # `seconds=None` explicitly disables trimming/interpolation, so it must survive the filter above.
        if "seconds" in overrides:
            values["seconds"] = overrides["seconds"]

        if "num_frames" not in values:
            if p.num_frames is not None:
                values["num_frames"] = p.num_frames
            else:
                seconds = values.get("seconds") or cls.seconds
                values["num_frames"] = frames_for_seconds(seconds, values.get("fps", cls.fps))
        return cls(**values)

    @property
    def gguf_file(self) -> str | None:
        return GGUF_FILE_TEMPLATE.format(quant=self.quant) if self.gguf_repo else None

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Prompt must not be empty.")
        for name in ("height", "width"):
            value = getattr(self, name)
            if value <= 0 or value % SPATIAL_MULTIPLE:
                raise ValueError(f"{name}={value} must be a positive multiple of {SPATIAL_MULTIPLE}.")
        if self.width * 9 != self.height * 16:
            raise ValueError(f"{self.width}x{self.height} is not 16:9.")
        if self.upscale not in UPSCALE_TARGETS:
            raise ValueError(f"upscale must be one of {', '.join(UPSCALE_TARGETS)}.")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError("seconds must be > 0.")
        if self.num_frames < 1 or (self.num_frames - 1) % TEMPORAL_MULTIPLE:
            raise ValueError(
                f"frames={self.num_frames} must be of the form 4k+1 (e.g. 17, 33, 49, 81)."
            )
        if self.num_frames > MAX_FRAMES:
            raise ValueError(f"frames={self.num_frames} exceeds the model's maximum of {MAX_FRAMES}.")
        if self.num_inference_steps < 1:
            raise ValueError("steps must be >= 1.")
        if self.device == "cpu" and self.offload != "none":
            raise ValueError("CPU offloading only applies to GPU devices; use --offload none with --device cpu.")
        if self.gguf_repo:
            if self.quant not in GGUF_QUANTS:
                raise ValueError(f"quant must be one of {', '.join(GGUF_QUANTS)}.")
            if self.device != "cuda":
                raise ValueError("GGUF (14B) models are only supported on CUDA; use --profile mac on Apple Silicon.")
        for path in self.reference_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Reference image not found: {path}")

    @property
    def target_frame_count(self) -> int:
        """Frames in the final video: exactly `seconds` at `fps`, or everything generated."""
        if self.seconds is None:
            return self.num_frames
        return round(self.seconds * self.fps)

    @property
    def raw_output_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.stem}_raw{self.output_path.suffix}")
