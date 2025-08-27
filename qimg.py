from typing import Optional
from PIL import Image
import torch

from diffusers import DiffusionPipeline
from diffusers import QwenImageEditPipeline
from diffusers import QwenImageTransformer2DModel
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model
import os

# -----------------------------
# Device and dtype configuration
# -----------------------------
if torch.cuda.is_available():
    _TORCH_DTYPE = torch.bfloat16
    _DEVICE = "cuda"
else:
    _TORCH_DTYPE = torch.float32
    _DEVICE = "cpu"

# Behavior toggles via environment
_ENABLE_CPU_OFFLOAD = os.environ.get("QIMG_CPU_OFFLOAD", "1") == "1"  # when on CUDA, offload model parts to CPU to reduce VRAM
_PERSIST_PIPELINES = os.environ.get("QIMG_PERSIST", "0") == "1"       # keep pipelines resident between calls

# -----------------------------
# Lazy singletons for pipelines
# -----------------------------
_img_pipe: Optional[DiffusionPipeline] = None
_edit_pipe: Optional[QwenImageEditPipeline] = None


def _get_img_pipeline() -> DiffusionPipeline:
    global _img_pipe
    if _img_pipe is None:
        model_name = "Qwen/Qwen-Image"
        # Build transformer in bfloat16 without initializing weights, then load DFloat11 weights
        with no_init_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                QwenImageTransformer2DModel.load_config(model_name, subfolder="transformer"),
            ).to(_TORCH_DTYPE)

        # Optional CPU offload controls via env
        cpu_offload = (_DEVICE == "cuda" and _ENABLE_CPU_OFFLOAD)
        blocks_env = os.environ.get("QIMG_CPU_OFFLOAD_BLOCKS")
        cpu_offload_blocks = int(blocks_env) if (blocks_env and blocks_env.isdigit()) else None
        pin_memory = os.environ.get("QIMG_PIN_MEMORY", "1") == "1"

        # Inject DFloat11 weights into the transformer
        DFloat11Model.from_pretrained(
            "DFloat11/Qwen-Image-DF11",
            device="cpu",
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory,
            bfloat16_model=transformer,
        )

        # Build the diffusion pipeline using the transformed transformer
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            transformer=transformer,
            torch_dtype=_TORCH_DTYPE,
        )
        # Memory optimizations
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        if _DEVICE == "cuda" and _ENABLE_CPU_OFFLOAD:
            # Reduce peak VRAM usage by offloading
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to(_DEVICE)
        else:
            pipe = pipe.to(_DEVICE)

        _img_pipe = pipe
    return _img_pipe


def _get_edit_pipeline() -> QwenImageEditPipeline:
    global _edit_pipe
    if _edit_pipe is None:
        model_name = "Qwen/Qwen-Image-Edit"
        # Build edit transformer in bf16 without initializing weights, then load DF11 edit weights
        with no_init_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                QwenImageTransformer2DModel.load_config(model_name, subfolder="transformer"),
            ).to(_TORCH_DTYPE)

        # Optional CPU offload controls via env
        cpu_offload = (_DEVICE == "cuda" and _ENABLE_CPU_OFFLOAD)
        blocks_env = os.environ.get("QIMG_CPU_OFFLOAD_BLOCKS")
        cpu_offload_blocks = int(blocks_env) if (blocks_env and blocks_env.isdigit()) else None
        pin_memory = os.environ.get("QIMG_PIN_MEMORY", "1") == "1"

        # Inject DFloat11 edit weights into the transformer
        DFloat11Model.from_pretrained(
            "DFloat11/Qwen-Image-Edit-DF11",
            device="cpu",
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory,
            bfloat16_model=transformer,
        )

        # Build the edit pipeline using the transformed transformer
        pipe = QwenImageEditPipeline.from_pretrained(
            model_name,
            transformer=transformer,
            torch_dtype=_TORCH_DTYPE,
        )
        # dtype and memory optimizations
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        if _DEVICE == "cuda" and _ENABLE_CPU_OFFLOAD:
            try:
                pipe = pipe.to(_TORCH_DTYPE)
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to(_TORCH_DTYPE).to(_DEVICE)
        else:
            pipe = pipe.to(_TORCH_DTYPE).to(_DEVICE)

        pipe.set_progress_bar_config(disable=None)
        _edit_pipe = pipe
    return _edit_pipe


# -----------------------------
# Public API
# -----------------------------


def img_generate(
    prompt: str,
    negative_prompt: str = " ",
    aspect_ratio: str = "1:1",
    steps: int = 50,
    true_cfg_scale: float = 4.0,
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image using Qwen/Qwen-Image.

    Parameters mirror the behavior in img.py with sensible defaults.
    Returns the PIL Image. If out_path is provided, the image is saved there.
    """
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    if aspect_ratio not in aspect_ratios:
        raise ValueError(f"Unsupported aspect_ratio: {aspect_ratio}. Choose from {sorted(aspect_ratios.keys())}")

    width, height = aspect_ratios[aspect_ratio]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_DEVICE).manual_seed(seed)

    pipe = _get_img_pipeline()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
    ).images[0]

    if out_path:
        image.save(out_path)

    return image


def img_edit(
    image: Image.Image,
    prompt: str,
    true_cfg_scale: float = 4.0,
    negative_prompt: str = "",
    steps: int = 100,
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Image.Image:
    """
    Edit an image using Qwen/Qwen-Image-Edit

    Accepts a PIL Image and a text prompt. Returns the edited PIL Image. If out_path
    is provided, the image is saved there.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    generator = None
    if seed is not None:
        # Qwen Image Edit uses PyTorch RNG; keep CPU generator acceptable across devices
        generator = torch.manual_seed(seed)

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
    }

    pipe = _get_edit_pipeline()

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]

    if out_path:
        output_image.save(out_path)

    return output_image


def unload_pipelines():
    """Release pipelines to free GPU/CPU memory. No-op if persistence is enabled.

    Controlled by env var QIMG_PERSIST (set to "1" to keep pipelines loaded).
    """
    global _img_pipe, _edit_pipe
    if _PERSIST_PIPELINES:
        return
    # Delete references and empty CUDA cache if applicable
    try:
        if _img_pipe is not None:
            del _img_pipe
    except Exception:
        pass
    try:
        if _edit_pipe is not None:
            del _edit_pipe
    except Exception:
        pass
    _img_pipe = None
    _edit_pipe = None
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
