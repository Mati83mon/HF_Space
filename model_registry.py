"""Task detection and task -> modality -> loader mapping.

This is the only module that knows which loader handles which task. Everything
else (UI, WebSocket dispatcher) goes through `resolve()` / `get_loader()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loaders.common import UnsupportedModelError, hf_token

MODALITIES = ("text", "image", "audio", "video")

# task -> modality. Keys match Hub `pipeline_tag` values wherever one exists.
TASK_MODALITY: Dict[str, str] = {
    # --- text ---
    "text-generation": "text",
    "text2text-generation": "text",
    "summarization": "text",
    "translation": "text",
    "text-classification": "text",
    "token-classification": "text",
    "question-answering": "text",
    "zero-shot-classification": "text",
    "fill-mask": "text",
    "feature-extraction": "text",
    "conversational": "text",
    # --- image ---
    "text-to-image": "image",
    "image-to-image": "image",
    "inpainting": "image",
    "image-classification": "image",
    "image-to-text": "image",
    # --- audio ---
    "text-to-speech": "audio",
    "text-to-audio": "audio",
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
    # --- video ---
    "text-to-video": "video",
    "image-to-video": "video",
    "image-to-gif": "video",
    "video-classification": "video",
}

# Tasks offered in the UI dropdowns, grouped per tab.
TASKS_BY_MODALITY: Dict[str, List[str]] = {
    modality: [task for task, mod in TASK_MODALITY.items() if mod == modality]
    for modality in MODALITIES
}

# Substring heuristics used when the Hub gives us nothing usable. Ordered:
# the first match wins, so put the most specific patterns first.
_NAME_HEURISTICS: List[tuple] = [
    (r"ltx-?video|cogvideo|hunyuanvideo|mochi|wan2|animatediff|text-?to-?video", "text-to-video"),
    (r"stable-video-diffusion|svd|img2vid|image-?to-?video", "image-to-video"),
    (r"pyramid-?flow|miniflux", "text-to-video"),
    (r"whisper|wav2vec2|parakeet|\basr\b", "automatic-speech-recognition"),
    (r"\btts\b|bark|speecht5|xtts|vits|kokoro|musicgen|audiogen", "text-to-speech"),
    (r"flux|stable-?diffusion|sdxl|playground-?v2|kandinsky|pixart|sana", "text-to-image"),
    (r"instruct|chat|llama|mistral|qwen|phi-|gemma|gpt", "text-generation"),
]

EXAMPLE_MODELS: Dict[str, List[str]] = {
    "text": [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    ],
    "image": [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        "runwayml/stable-diffusion-v1-5",
    ],
    "audio": [
        "openai/whisper-small",
        "microsoft/speecht5_tts",
        "suno/bark-small",
        "facebook/musicgen-small",
    ],
    "video": [
        "Lightricks/LTX-Video",
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "THUDM/CogVideoX-2b",
        "tencent/HunyuanVideo",
    ],
}


@dataclass
class ModelSpec:
    """Result of resolving a repo_id into something a loader can act on."""

    repo_id: str
    task: str
    modality: str
    library: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: str = "hub"  # hub | manual | heuristic
    note: str = ""

    def summary(self) -> str:
        lines = [
            f"repo_id  : {self.repo_id}",
            f"task     : {self.task}  (detected via: {self.source})",
            f"modality : {self.modality}",
        ]
        if self.library:
            lines.append(f"library  : {self.library}")
        if self.tags:
            lines.append(f"tags     : {', '.join(self.tags[:12])}")
        if self.note:
            lines.append(f"note     : {self.note}")
        return "\n".join(lines)


def fetch_model_info(repo_id: str) -> Dict[str, Any]:
    """Query the Hub for metadata. Returns {} when the Hub is unreachable."""
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id, token=hf_token())
    except Exception as exc:  # offline Space, gated repo, typo — all non-fatal here
        return {"error": str(exc)}

    return {
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "tags": list(getattr(info, "tags", []) or []),
    }


def _guess_from_name_and_tags(repo_id: str, tags: List[str], library: Optional[str]) -> Optional[str]:
    haystack = " ".join([repo_id] + list(tags)).lower()

    # Explicit task tags beat fuzzy name matching.
    for tag in tags:
        if tag in TASK_MODALITY:
            return tag

    for pattern, task in _NAME_HEURISTICS:
        if re.search(pattern, haystack):
            return task

    if library == "diffusers":
        return "text-to-image"
    if library in ("transformers", "sentence-transformers"):
        return "text-generation"
    return None


def resolve(repo_id: str, task_override: str = "auto") -> ModelSpec:
    """Turn a repo_id (+ optional manual task) into a ModelSpec.

    Precedence: manual override > Hub pipeline_tag > tag/name heuristics.
    """
    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise UnsupportedModelError("No repo_id given — paste something like 'Qwen/Qwen2.5-1.5B-Instruct'.")

    override = (task_override or "auto").strip()
    if override and override != "auto":
        if override not in TASK_MODALITY:
            raise UnsupportedModelError(
                f"Unknown task '{override}'. Supported: {', '.join(sorted(TASK_MODALITY))}"
            )
        return ModelSpec(
            repo_id=repo_id,
            task=override,
            modality=TASK_MODALITY[override],
            source="manual",
        )

    info = fetch_model_info(repo_id)
    tags = info.get("tags", []) or []
    library = info.get("library_name")
    pipeline_tag = info.get("pipeline_tag")

    if pipeline_tag and pipeline_tag in TASK_MODALITY:
        return ModelSpec(
            repo_id=repo_id,
            task=pipeline_tag,
            modality=TASK_MODALITY[pipeline_tag],
            library=library,
            tags=tags,
            source="hub",
        )

    guess = _guess_from_name_and_tags(repo_id, tags, library)
    if guess:
        note = "Hub metadata was inconclusive; guessed from tags/name."
        if info.get("error"):
            note = f"Hub lookup failed ({info['error'][:120]}); guessed from name."
        return ModelSpec(
            repo_id=repo_id,
            task=guess,
            modality=TASK_MODALITY[guess],
            library=library,
            tags=tags,
            source="heuristic",
            note=note,
        )

    raise UnsupportedModelError(
        f"Could not work out what '{repo_id}' does"
        + (f" (pipeline_tag={pipeline_tag!r})" if pipeline_tag else "")
        + ". Set 'Task override' manually in the tab you are using."
    )


def get_loader(modality: str):
    """Import and return the loader module for a modality (lazy on purpose)."""
    if modality == "text":
        from loaders import text as module
    elif modality == "image":
        from loaders import image as module
    elif modality == "audio":
        from loaders import audio as module
    elif modality == "video":
        from loaders import video as module
    else:
        raise UnsupportedModelError(f"No loader registered for modality '{modality}'.")
    return module


def load(repo_id: str, task_override: str = "auto", **options):
    """Convenience: resolve + load in one call. Returns (ModelHandle, ModelSpec)."""
    spec = resolve(repo_id, task_override)
    loader = get_loader(spec.modality)
    handle = loader.load_model(spec.repo_id, spec.task, **options)
    return handle, spec
