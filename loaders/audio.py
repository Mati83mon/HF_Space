"""Audio loader: TTS / text-to-audio and ASR / audio-classification."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from loaders.common import (
    ModelHandle,
    UnsupportedModelError,
    cached_load,
    get_device,
    hf_token,
    require,
)

TTS_TASKS = {"text-to-speech", "text-to-audio"}
ASR_TASKS = {"automatic-speech-recognition"}


def load_model(repo_id: str, task: str = "text-to-speech", dtype: str = "auto", **_: Any) -> ModelHandle:
    key = ("audio", repo_id, task, dtype)

    def _factory() -> ModelHandle:
        transformers = require("transformers")
        device = get_device()

        kwargs: Dict[str, Any] = {"token": hf_token()}
        if device == "cuda":
            kwargs["device"] = 0

        try:
            pipe = transformers.pipeline(task, model=repo_id, **kwargs)
        except Exception as exc:
            raise UnsupportedModelError(
                f"transformers could not build a '{task}' pipeline for {repo_id}: {exc}"
            ) from exc

        return ModelHandle(
            repo_id=repo_id,
            task=task,
            modality="audio",
            pipeline=pipe,
            meta={"device": device, "cls": type(pipe).__name__},
        )

    return cached_load(key, _factory)


def run_inference(
    handle: ModelHandle,
    text: str = "",
    audio_input: Any = None,
    speaker_embedding: Optional[str] = None,
    **params: Any,
):
    """Return (audio_tuple_or_None, text_output).

    `audio_tuple` is Gradio's `(sample_rate, numpy_array)` shape so the UI can
    play it back directly.
    """
    pipe = handle.pipeline
    if pipe is None:
        raise UnsupportedModelError("Model was unloaded; press Load again.")

    if handle.task in TTS_TASKS:
        if not text:
            raise UnsupportedModelError("TTS needs some input text.")
        forward = _speaker_kwargs(handle, speaker_embedding)
        result = pipe(text, **forward)
        audio, rate = _unpack_audio(result)
        return (rate, audio), f"Synthesised {len(audio) / max(rate, 1):.2f}s @ {rate} Hz"

    if handle.task in ASR_TASKS:
        if audio_input is None:
            raise UnsupportedModelError("ASR needs an audio file or a microphone recording.")
        kwargs: Dict[str, Any] = {}
        if params.get("return_timestamps"):
            kwargs["return_timestamps"] = True
        if params.get("language"):
            kwargs["generate_kwargs"] = {"language": params["language"]}
        result = pipe(_as_asr_input(audio_input), **kwargs)
        if isinstance(result, dict):
            return None, str(result.get("text", result))
        return None, str(result)

    # audio-classification and anything else that returns records
    import json

    result = pipe(_as_asr_input(audio_input) if audio_input is not None else text)
    return None, json.dumps(result, indent=2, ensure_ascii=False, default=str)


def _speaker_kwargs(handle: ModelHandle, speaker_embedding: Optional[str]) -> Dict[str, Any]:
    """SpeechT5-style models need an x-vector; everything else needs nothing."""
    model_name = handle.repo_id.lower()
    if "speecht5" not in model_name:
        return {}
    try:
        datasets = require("datasets")
        torch = require("torch")
        embeddings = datasets.load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        index = int(speaker_embedding) if speaker_embedding else 7306
        vector = torch.tensor(embeddings[index]["xvector"]).unsqueeze(0)
        return {"forward_params": {"speaker_embeddings": vector}}
    except Exception as exc:
        raise UnsupportedModelError(
            f"{handle.repo_id} needs speaker embeddings and they could not be fetched: {exc}"
        ) from exc


def _unpack_audio(result: Any) -> Tuple[Any, int]:
    """Normalise a TTS pipeline result into (1-D array, sample_rate)."""
    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict):
        raise UnsupportedModelError(f"Unexpected TTS output: {type(result).__name__}")

    audio = result.get("audio")
    rate = int(result.get("sampling_rate", 16000))
    if audio is None:
        raise UnsupportedModelError("TTS pipeline returned no audio.")

    try:
        import numpy as np

        audio = np.asarray(audio)
        audio = np.squeeze(audio)
        if audio.ndim > 1:  # (channels, samples) -> mono-first layout for Gradio
            audio = audio.T
    except ImportError:
        pass
    return audio, rate


def _as_asr_input(audio_input: Any) -> Any:
    """Accept a filepath, or Gradio's (sample_rate, ndarray) tuple."""
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        rate, array = audio_input
        try:
            import numpy as np

            array = np.asarray(array).astype("float32")
            peak = float(np.max(np.abs(array))) if array.size else 0.0
            if peak > 1.0:  # int16 input coming from the microphone
                array = array / 32768.0
            if array.ndim > 1:
                array = array.mean(axis=1)
        except ImportError:
            pass
        return {"array": array, "sampling_rate": int(rate)}
    return audio_input
