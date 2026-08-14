"""Image loader: Stable Diffusion / SDXL / FLUX / any Diffusers text2img pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from loaders.common import (
    ModelHandle,
    UnsupportedModelError,
    cached_load,
    get_device,
    get_dtype,
    hf_token,
    require,
    resolve_seed,
)

DIFFUSION_TASKS = {"text-to-image", "image-to-image", "inpainting"}


def load_model(
    repo_id: str,
    task: str = "text-to-image",
    dtype: str = "auto",
    uncensored: bool = False,
    **_: Any,
) -> ModelHandle:
    """Load (or reuse) an image pipeline.

    `uncensored` only controls Diffusers' optional post-hoc safety checker; it is
    part of the cache key because the two variants are different objects.
    """
    key = ("image", repo_id, task, dtype, bool(uncensored))

    def _factory() -> ModelHandle:
        if task not in DIFFUSION_TASKS:
            return _load_transformers_image(repo_id, task, dtype)

        diffusers = require("diffusers")
        device = get_device()
        torch_dtype = get_dtype(dtype)

        kwargs: Dict[str, Any] = {"token": hf_token()}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if uncensored:
            # Do not add our own filtering, and skip the optional one when the
            # user explicitly asked for raw model output.
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        auto_cls = {
            "text-to-image": "AutoPipelineForText2Image",
            "image-to-image": "AutoPipelineForImage2Image",
            "inpainting": "AutoPipelineForInpainting",
        }[task]

        pipe = _from_pretrained(diffusers, auto_cls, repo_id, kwargs)
        pipe = _place(pipe, device)

        return ModelHandle(
            repo_id=repo_id,
            task=task,
            modality="image",
            pipeline=pipe,
            meta={
                "device": device,
                "dtype": str(torch_dtype),
                "cls": type(pipe).__name__,
                "uncensored": uncensored,
            },
        )

    return cached_load(key, _factory)


def _from_pretrained(diffusers, auto_cls: str, repo_id: str, kwargs: Dict[str, Any]):
    """Try the Auto* pipeline first, then the generic DiffusionPipeline."""
    errors = []
    for cls_name in (auto_cls, "DiffusionPipeline"):
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(repo_id, **kwargs)
        except Exception as exc:
            errors.append(f"{cls_name}: {exc}")
            # safety_checker=None is rejected by pipelines that have no such arg
            if "safety_checker" in kwargs:
                retry = {k: v for k, v in kwargs.items() if k not in ("safety_checker", "requires_safety_checker")}
                try:
                    return cls.from_pretrained(repo_id, **retry)
                except Exception as exc2:
                    errors.append(f"{cls_name} (no safety args): {exc2}")
    raise UnsupportedModelError(
        f"Diffusers could not load {repo_id}.\n" + "\n".join(errors[:4])
    )


def _place(pipe, device: str):
    """Move the pipeline to the device, preferring CPU offload on CUDA."""
    if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
            return pipe
        except Exception:
            pass
    try:
        return pipe.to(device)
    except Exception:
        return pipe


def _load_transformers_image(repo_id: str, task: str, dtype: str) -> ModelHandle:
    """image-classification / image-to-text run through transformers."""
    transformers = require("transformers")
    device = get_device()
    try:
        pipe = transformers.pipeline(task, model=repo_id, token=hf_token())
    except Exception as exc:
        raise UnsupportedModelError(f"Could not build '{task}' pipeline for {repo_id}: {exc}") from exc
    return ModelHandle(
        repo_id=repo_id,
        task=task,
        modality="image",
        pipeline=pipe,
        meta={"device": device, "cls": type(pipe).__name__},
    )


def run_inference(
    handle: ModelHandle,
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 25,
    guidance_scale: float = 7.0,
    width: int = 768,
    height: int = 768,
    seed: int = -1,
    num_images: int = 1,
    init_image: Any = None,
    mask_image: Any = None,
    strength: float = 0.6,
    **extra: Any,
):
    """Run the pipeline. Returns (images, info_string)."""
    pipe = handle.pipeline
    if pipe is None:
        raise UnsupportedModelError("Model was unloaded; press Load again.")

    if handle.task not in DIFFUSION_TASKS:
        import json

        result = pipe(init_image if init_image is not None else prompt)
        return [], json.dumps(result, indent=2, ensure_ascii=False, default=str)

    generator, used_seed = resolve_seed(seed)

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "num_images_per_prompt": int(num_images),
    }
    if generator is not None:
        kwargs["generator"] = generator
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    # FLUX & co. use `guidance_scale` differently but still accept it; schnell-style
    # distilled models ignore it, which is harmless.
    kwargs["guidance_scale"] = float(guidance_scale)

    if handle.task == "text-to-image":
        kwargs["width"] = int(width)
        kwargs["height"] = int(height)
    else:
        if init_image is None:
            raise UnsupportedModelError(f"Task '{handle.task}' requires an input image.")
        kwargs["image"] = init_image
        kwargs["strength"] = float(strength)
        if handle.task == "inpainting":
            if mask_image is None:
                raise UnsupportedModelError("Inpainting requires a mask image.")
            kwargs["mask_image"] = mask_image

    kwargs.update({k: v for k, v in extra.items() if v is not None})
    kwargs = _drop_unsupported(pipe, kwargs)

    result = pipe(**kwargs)
    images = getattr(result, "images", result)
    info = (
        f"{handle.repo_id} | steps={steps} cfg={guidance_scale} "
        f"size={width}x{height} seed={used_seed if used_seed is not None else 'random'}"
    )
    return list(images), info


def _drop_unsupported(pipe, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs down to what this specific pipeline's __call__ accepts."""
    import inspect

    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in signature.parameters}
