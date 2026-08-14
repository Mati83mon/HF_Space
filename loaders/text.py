"""Text loader: causal LMs, seq2seq, classification and friends (transformers)."""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, Optional

from loaders.common import (
    ModelHandle,
    UnsupportedModelError,
    cached_load,
    dtype_kwarg,
    get_device,
    get_dtype,
    hf_token,
    require,
)

GENERATIVE_TASKS = {"text-generation", "text2text-generation", "summarization", "translation", "conversational"}


def load_model(repo_id: str, task: str = "text-generation", dtype: str = "auto", **_: Any) -> ModelHandle:
    """Load (or reuse) a transformers pipeline for `repo_id`."""
    key = ("text", repo_id, task, dtype)

    def _factory() -> ModelHandle:
        transformers = require("transformers")
        device = get_device()
        torch_dtype = get_dtype(dtype)

        # `conversational` was removed as a pipeline task; chat models are just
        # causal LMs with a chat template applied on top.
        pipeline_task = "text-generation" if task == "conversational" else task

        kwargs: Dict[str, Any] = {"token": hf_token()}
        if torch_dtype is not None:
            # transformers renamed `torch_dtype` to `dtype` in v5.
            kwargs[dtype_kwarg(transformers)] = torch_dtype
        if device == "cuda":
            kwargs["device_map"] = "auto"

        try:
            pipe = transformers.pipeline(pipeline_task, model=repo_id, **kwargs)
        except Exception as exc:
            raise UnsupportedModelError(
                f"transformers could not build a '{pipeline_task}' pipeline for {repo_id}: {exc}"
            ) from exc

        return ModelHandle(
            repo_id=repo_id,
            task=task,
            modality="text",
            pipeline=pipe,
            meta={"device": device, "dtype": str(torch_dtype), "cls": type(pipe).__name__},
        )

    return cached_load(key, _factory)


def _generation_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    temperature = float(params.get("temperature", 0.7))
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(params.get("max_new_tokens", 256)),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = float(params.get("top_p", 0.95))
        top_k = int(params.get("top_k", 0))
        if top_k > 0:
            kwargs["top_k"] = top_k
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


def parse_stop_sequences(raw: str) -> List[str]:
    """Comma-separated UI field -> list of stop strings ('' entries dropped)."""
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _apply_stops(text: str, stops: List[str]) -> str:
    cut = len(text)
    for stop in stops:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def render_chat(handle: ModelHandle, messages: List[Dict[str, str]]) -> str:
    """Render chat messages with the model's own template when it has one.

    Falls back to a plain `ROLE: content` transcript so that base models (which
    ship no chat template) still get something sensible.
    """
    tokenizer = getattr(handle.pipeline, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None) if tokenizer else None
    if tokenizer is not None and template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass  # malformed roles (e.g. a tool message the template rejects)
    lines = [f"{m['role'].upper()}: {m['content']}" for m in messages if m.get("content")]
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def run_inference(handle: ModelHandle, prompt: str, **params: Any) -> str:
    """Single-shot inference. Returns text ready to display."""
    pipe = handle.pipeline
    if pipe is None:
        raise UnsupportedModelError("Model was unloaded; press Load again.")

    if handle.task in GENERATIVE_TASKS:
        kwargs = _generation_kwargs(params)
        if handle.task in ("text-generation", "conversational"):
            kwargs["return_full_text"] = False
        outputs = pipe(prompt, **kwargs)
        text = _first_text(outputs)
        return _apply_stops(text, params.get("stop_sequences") or [])

    if handle.task == "zero-shot-classification":
        labels = params.get("candidate_labels") or ["positive", "negative"]
        return _format_records(pipe(prompt, candidate_labels=labels))

    if handle.task == "question-answering":
        context = params.get("context") or ""
        if not context:
            raise UnsupportedModelError("question-answering needs a 'context' parameter.")
        return _format_records(pipe(question=prompt, context=context))

    # classification / token-classification / fill-mask / feature-extraction
    return _format_records(pipe(prompt))


def stream_inference(handle: ModelHandle, prompt: str, **params: Any) -> Iterator[str]:
    """Yield the growing completion token by token (generative tasks only)."""
    if handle.task not in GENERATIVE_TASKS:
        yield run_inference(handle, prompt, **params)
        return

    transformers = require("transformers")
    pipe = handle.pipeline
    tokenizer = pipe.tokenizer
    streamer = transformers.TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    kwargs = _generation_kwargs(params)
    kwargs["streamer"] = streamer
    if handle.task in ("text-generation", "conversational"):
        kwargs["return_full_text"] = False

    error: Dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            pipe(prompt, **kwargs)
        except BaseException as exc:  # surfaced after the loop below drains
            error["exc"] = exc
            streamer.end()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    stops = params.get("stop_sequences") or []
    acc = ""
    for chunk in streamer:
        acc += chunk
        trimmed = _apply_stops(acc, stops)
        yield trimmed
        if trimmed != acc:  # a stop sequence landed — stop pulling tokens
            break

    thread.join(timeout=1.0)
    if "exc" in error:
        raise UnsupportedModelError(f"Generation failed: {error['exc']}")


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #


def _first_text(outputs: Any) -> str:
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            for key in ("generated_text", "summary_text", "translation_text"):
                if key in first:
                    return str(first[key])
        return str(first)
    if isinstance(outputs, dict):
        return str(outputs.get("generated_text", outputs))
    return str(outputs)


def _format_records(outputs: Any) -> str:
    import json

    try:
        return json.dumps(outputs, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(outputs)
