from __future__ import annotations

import gc
import hashlib
import json
import logging
import time

import numpy as np
import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanVACEPipeline,
)
from diffusers.utils import export_to_video

from video_gen.config import GenConfig
from video_gen.postprocess import needs_postprocess, postprocess
from video_gen.utils import format_duration, load_reference_images

logger = logging.getLogger(__name__)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _embeds_cache_key(cfg: GenConfig, use_cfg: bool) -> str:
    payload = json.dumps(
        {
            "model": cfg.model_id,
            "prompt": cfg.prompt,
            "negative": cfg.negative_prompt if use_cfg else None,
            "max_len": cfg.max_sequence_length,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_text_pipeline(cfg: GenConfig) -> WanVACEPipeline:
    dtype = DTYPES[cfg.text_encoder_dtype]
    if cfg.text_encoder_device == "cpu":
        # Stored in bfloat16 on the Hub; loading in the same dtype on CPU avoids an ~11 GB conversion copy.
        return WanVACEPipeline.from_pretrained(cfg.model_id, transformer=None, vae=None, torch_dtype=dtype)

    from transformers import UMT5EncoderModel

    # device_map streams each shard straight to the GPU, so the 11 GB encoder never sits in system RAM.
    # UMT5 keeps its `wo` projections in float32, which keeps float16 inference free of overflows.
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.model_id, subfolder="text_encoder", dtype=dtype, device_map=cfg.text_encoder_device
    )
    return WanVACEPipeline.from_pretrained(cfg.model_id, text_encoder=text_encoder, transformer=None, vae=None)


def encode_prompts(cfg: GenConfig) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the UMT5-XXL text encoder, then release it before the transformer is loaded.

    The encoder is ~11 GB, so it never shares memory with the denoiser. Results are cached on
    disk so repeated prompts skip this step entirely.
    """
    use_cfg = cfg.guidance_scale > 1.0
    cache_path = cfg.embeds_cache_dir / f"{_embeds_cache_key(cfg, use_cfg)}.pt"
    if cache_path.is_file():
        logger.info("Using cached prompt embeddings: %s", cache_path)
        cached = torch.load(cache_path, map_location="cpu")
        return cached["prompt_embeds"], cached["negative_prompt_embeds"]

    logger.info("Loading text encoder on %s (first run downloads ~11 GB)...", cfg.text_encoder_device)
    start = time.perf_counter()
    text_pipe = _load_text_pipeline(cfg)
    logger.info("Encoding prompt...")
    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = text_pipe.encode_prompt(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            do_classifier_free_guidance=use_cfg,
            max_sequence_length=cfg.max_sequence_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    del text_pipe
    _free_memory()
    logger.info("Prompt encoded in %s", format_duration(time.perf_counter() - start))

    cfg.embeds_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds},
        cache_path,
    )
    return prompt_embeds, negative_prompt_embeds


def _maybe_patch_conv3d(cfg: GenConfig) -> None:
    if not cfg.use_conv3d_patch:
        return
    try:
        from mps_conv3d import patch_conv3d
    except ImportError as exc:
        raise RuntimeError(
            "--conv3d-patch requires the optional dependency: uv sync --extra conv3d"
        ) from exc
    patch_conv3d()
    logger.info("Patched F.conv3d with native Metal kernel (mps-conv3d).")


def _load_gguf_transformer(cfg: GenConfig):
    from diffusers import GGUFQuantizationConfig, WanVACETransformer3DModel
    from huggingface_hub import hf_hub_download

    logger.info("Downloading/locating %s from %s...", cfg.gguf_file, cfg.gguf_repo)
    path = hf_hub_download(cfg.gguf_repo, cfg.gguf_file)
    dtype = DTYPES[cfg.dtype]
    logger.info("Loading 14B transformer (%s, compute %s) onto %s...", cfg.quant, cfg.dtype, cfg.device)
    return WanVACETransformer3DModel.from_single_file(
        path,
        config=cfg.model_id,
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        device=cfg.device,
    )


def build_pipeline(cfg: GenConfig) -> WanVACEPipeline:
    """Load transformer + VAE without the text encoder."""
    _maybe_patch_conv3d(cfg)
    dtype = DTYPES[cfg.dtype]

    transformer = _load_gguf_transformer(cfg) if cfg.gguf_repo else None

    logger.info("Loading %sVAE (float32)...", "" if transformer is not None else f"transformer ({cfg.dtype}) and ")
    # The VAE must stay float32: half-precision decoding produces artifacts.
    vae = AutoencoderKLWan.from_pretrained(cfg.model_id, subfolder="vae", torch_dtype=torch.float32)
    components = {"vae": vae, "text_encoder": None, "tokenizer": None}
    if transformer is not None:
        components["transformer"] = transformer
    pipe = WanVACEPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype, **components)

    if cfg.scheduler == "euler":
        # The LightX2V distillation was trained with Euler flow-matching steps.
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=cfg.flow_shift)
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=cfg.flow_shift)

    if cfg.vae_tiling:
        pipe.vae.enable_tiling()

    if cfg.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=cfg.device)
    elif cfg.offload == "model":
        pipe.enable_model_cpu_offload(device=cfg.device)
    elif transformer is not None:
        # The GGUF transformer was loaded straight onto the device.
        pipe.vae.to(cfg.device)
    else:
        pipe.to(cfg.device)

    pipe.set_progress_bar_config(desc="Denoising")
    return pipe


def _decode_latents(
    vae: AutoencoderKLWan, video_processor, latents: torch.Tensor, num_reference_images: int
) -> np.ndarray:
    """Same decoding as WanVACEPipeline, run after the transformer has been freed."""
    latents = latents[:, :, num_reference_images:].to(vae.dtype)
    shape = (1, vae.config.z_dim, 1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean).view(shape).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(shape).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    return video_processor.postprocess_video(video, output_type="np")[0]


def _check_device(cfg: GenConfig) -> None:
    if cfg.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine; use --profile t4 on CUDA or --device cpu.")
    if "cuda" in (cfg.device, cfg.text_encoder_device) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; on Colab choose Runtime > Change runtime type > T4 GPU.")


def generate(cfg: GenConfig) -> None:
    cfg.validate()
    _check_device(cfg)

    total_start = time.perf_counter()
    references = load_reference_images(cfg.reference_paths, cfg.height, cfg.width)
    if not references:
        logger.warning("No reference images given; generating from the prompt alone.")

    prompt_embeds, negative_prompt_embeds = encode_prompts(cfg)

    pipe = build_pipeline(cfg)
    # A CPU generator keeps seeds reproducible across devices.
    generator = torch.Generator(device="cpu")
    seed = cfg.seed if cfg.seed is not None else generator.seed()
    generator.manual_seed(seed)

    # On a fully resident GPU model, decode separately so the VAE gets the transformer's memory.
    decode_separately = cfg.device == "cuda" and cfg.offload == "none"

    logger.info(
        "Generating %d frames at %dx%d, %d steps, %d reference image(s), seed=%d",
        cfg.num_frames, cfg.width, cfg.height, cfg.num_inference_steps, len(references), seed,
    )
    denoise_start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            reference_images=references or None,
            conditioning_scale=cfg.conditioning_scale,
            height=cfg.height,
            width=cfg.width,
            num_frames=cfg.num_frames,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            output_type="latent" if decode_separately else "np",
        )

        if decode_separately:
            latents, vae, video_processor = result.frames, pipe.vae, pipe.video_processor
            del result, pipe
            _free_memory()
            logger.info("Decoding latents with the VAE...")
            frames = _decode_latents(vae, video_processor, latents, len(references))
            del latents, vae
        else:
            frames = result.frames[0]
            del result, pipe
    _free_memory()
    logger.info("Denoising + decoding took %s", format_duration(time.perf_counter() - denoise_start))

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    if needs_postprocess(cfg):
        export_to_video(frames, str(cfg.raw_output_path), fps=cfg.fps)
        logger.info("Saved raw %dx%d clip: %s", cfg.width, cfg.height, cfg.raw_output_path)
        postprocess(frames, cfg)
    else:
        export_to_video(frames, str(cfg.output_path), fps=cfg.fps)
    logger.info("Saved %s (total %s)", cfg.output_path, format_duration(time.perf_counter() - total_start))
