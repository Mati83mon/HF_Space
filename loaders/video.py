"""Video loader: text-to-video, image-to-video and image-to-gif via Diffusers.

Covers LTX-Video, Stable Video Diffusion, CogVideoX, HunyuanVideo, Pyramid-Flow
miniFLUX and FLUX-family video pipelines. Rather than hard-coding a class per
repo, we ask Diffusers to resolve the pipeline from the repo's `model_index.json`
and only fall back to explicit classes for repos the Auto* map does not know.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loaders.common import (
    ModelHandle,
    UnsupportedModelError,
    cached_load,
    get_device,
    get_dtype,
    hf_token,
    require,
    resolve_seed,
    temp_path,
)

VIDEO_TASKS = {"text-to-video", "image-to-video", "image-to-gif"}

# Repos whose pipeline class is worth naming explicitly (DiffusionPipeline can
# resolve most of these on its own, but being explicit gives better errors).
KNOWN_PIPELINES: Dict[str, str] = {
    "lightricks/ltx-video": "LTXPipeline",
    "stabilityai/stable-video-diffusion-img2vid": "StableVideoDiffusionPipeline",
    "stabilityai/stable-video-diffusion-img2vid-xt": "StableVideoDiffusionPipeline",
    "thudm/cogvideox-2b": "CogVideoXPipeline",
    "thudm/cogvideox-5b": "CogVideoXPipeline",
    "tencent/hunyuanvideo": "HunyuanVideoPipeline",
}

# Same repos, image-to-video variants.
KNOWN_I2V_PIPELINES: Dict[str, str] = {
    "lightricks/ltx-video": "LTXImageToVideoPipeline",
    "thudm/cogvideox-5b-i2v": "CogVideoXImageToVideoPipeline",
    "stabilityai/stable-video-diffusion-img2vid": "StableVideoDiffusionPipeline",
    "stabilityai/stable-video-diffusion-img2vid-xt": "StableVideoDiffusionPipeline",
}


def load_model(
    repo_id: str,
    task: str = "text-to-video",
    dtype: str = "auto",
    uncensored: bool = False,
    **_: Any,
) -> ModelHandle:
    key = ("video", repo_id, task, dtype, bool(uncensored))

    def _factory() -> ModelHandle:
        if task not in VIDEO_TASKS:
            return _load_classifier(repo_id, task)

        diffusers = require("diffusers")
        device = get_device()
        # Video pipelines are the memory-hungriest thing here; bf16 is the norm.
        torch_dtype = get_dtype(dtype if dtype != "auto" else ("bfloat16" if device == "cuda" else "float32"))

        kwargs: Dict[str, Any] = {"token": hf_token()}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        pipe = _build_pipeline(diffusers, repo_id, task, kwargs)
        pipe = _place(pipe, device)

        return ModelHandle(
            repo_id=repo_id,
            task=task,
            modality="video",
            pipeline=pipe,
            meta={
                "device": device,
                "dtype": str(torch_dtype),
                "cls": type(pipe).__name__,
                "uncensored": uncensored,
            },
        )

    return cached_load(key, _factory)


def _build_pipeline(diffusers, repo_id: str, task: str, kwargs: Dict[str, Any]):
    lookup = KNOWN_I2V_PIPELINES if task != "text-to-video" else KNOWN_PIPELINES
    candidates: List[str] = []

    named = lookup.get(repo_id.lower())
    if named:
        candidates.append(named)
    candidates.append("DiffusionPipeline")

    errors = []
    for cls_name in candidates:
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            errors.append(f"{cls_name}: not available in this diffusers version")
            continue
        try:
            return cls.from_pretrained(repo_id, **kwargs)
        except Exception as exc:
            errors.append(f"{cls_name}: {exc}")

    raise UnsupportedModelError(
        f"Could not load video pipeline for {repo_id} (task={task}).\n"
        + "\n".join(errors[:4])
        + "\nHint: some video repos need a newer diffusers or a custom pipeline — "
          "add it to loaders/video.py::KNOWN_PIPELINES."
    )


def _place(pipe, device: str):
    """Prefer sequential/model CPU offload — video pipelines rarely fit in VRAM."""
    if device == "cuda":
        for method in ("enable_model_cpu_offload", "enable_sequential_cpu_offload"):
            if hasattr(pipe, method):
                try:
                    getattr(pipe, method)()
                    _enable_vae_tricks(pipe)
                    return pipe
                except Exception:
                    continue
    try:
        pipe = pipe.to(device)
    except Exception:
        pass
    _enable_vae_tricks(pipe)
    return pipe


def _enable_vae_tricks(pipe) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    for method in ("enable_slicing", "enable_tiling"):
        if hasattr(vae, method):
            try:
                getattr(vae, method)()
            except Exception:
                pass


def _load_classifier(repo_id: str, task: str) -> ModelHandle:
    transformers = require("transformers")
    try:
        pipe = transformers.pipeline(task, model=repo_id, token=hf_token())
    except Exception as exc:
        raise UnsupportedModelError(f"Could not build '{task}' pipeline for {repo_id}: {exc}") from exc
    return ModelHandle(
        repo_id=repo_id,
        task=task,
        modality="video",
        pipeline=pipe,
        meta={"device": get_device(), "cls": type(pipe).__name__},
    )


def run_inference(
    handle: ModelHandle,
    prompt: str = "",
    negative_prompt: str = "",
    init_image: Any = None,
    num_frames: int = 49,
    fps: int = 8,
    steps: int = 30,
    guidance_scale: float = 6.0,
    width: int = 704,
    height: int = 480,
    seed: int = -1,
    output_format: str = "mp4",
    **extra: Any,
):
    """Run a video pipeline and export the result. Returns (path, info)."""
    pipe = handle.pipeline
    if pipe is None:
        raise UnsupportedModelError("Model was unloaded; press Load again.")

    if handle.task not in VIDEO_TASKS:
        import json

        return None, json.dumps(pipe(init_image), indent=2, ensure_ascii=False, default=str)

    if handle.task != "text-to-video" and init_image is None:
        raise UnsupportedModelError(f"Task '{handle.task}' requires an input image.")

    generator, used_seed = resolve_seed(seed)

    kwargs: Dict[str, Any] = {
        "num_inference_steps": int(steps),
        "num_frames": int(num_frames),
        "guidance_scale": float(guidance_scale),
        "width": int(width),
        "height": int(height),
    }
    if prompt:
        kwargs["prompt"] = prompt
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if init_image is not None:
        kwargs["image"] = init_image
    if generator is not None:
        kwargs["generator"] = generator

    kwargs.update({k: v for k, v in extra.items() if v is not None})
    kwargs = _drop_unsupported(pipe, kwargs)

    result = pipe(**kwargs)
    frames = _extract_frames(result)

    path = _export(frames, fps=int(fps), output_format=output_format)
    info = (
        f"{handle.repo_id} | {len(frames)} frames @ {fps} fps | steps={steps} "
        f"| {width}x{height} | seed={used_seed if used_seed is not None else 'random'}"
    )
    return path, info


def _extract_frames(result: Any):
    frames = getattr(result, "frames", None)
    if frames is None:
        frames = result
    # Diffusers returns a batch: frames[0] is the first video's frame list.
    if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
        return frames[0]
    try:  # numpy batch of shape (batch, frames, h, w, c)
        import numpy as np

        if isinstance(frames, np.ndarray) and frames.ndim == 5:
            return list(frames[0])
    except ImportError:
        pass
    return list(frames)


def _export(frames, fps: int, output_format: str) -> str:
    """Write frames to .mp4 or .gif and return the file path."""
    diffusers_utils = require("diffusers").utils

    if output_format == "gif":
        path = temp_path(".gif")
        diffusers_utils.export_to_gif(frames, path, fps=fps)
        return path

    path = temp_path(".mp4")
    try:
        diffusers_utils.export_to_video(frames, path, fps=fps)
    except Exception as exc:
        # imageio/av backends are the usual culprit — GIF always works.
        gif_path = temp_path(".gif")
        diffusers_utils.export_to_gif(frames, gif_path, fps=fps)
        raise UnsupportedModelError(
            f"MP4 export failed ({exc}); a GIF was written to {gif_path} instead. "
            "Install 'imageio[ffmpeg]' for MP4 output."
        )
    return path


def _drop_unsupported(pipe, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in signature.parameters}
