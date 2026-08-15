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
        result = pipe(_as_asr_input(audio_input, _target_sr(pipe)), **kwargs)
        if isinstance(result, dict):
            return None, str(result.get("text", result))
        return None, str(result)

    # audio-classification and anything else that returns records
    import json

    result = pipe(_as_asr_input(audio_input, _target_sr(pipe)) if audio_input is not None else text)
    return None, json.dumps(result, indent=2, ensure_ascii=False, default=str)


_XVECTOR_REPO = "Matthijs/cmu-arctic-xvectors"
_XVECTOR_ARCHIVE = "spkrec-xvect.zip"
_XVECTOR_CACHE: Dict[int, Any] = {}


def _speaker_kwargs(handle: ModelHandle, speaker_embedding: Optional[str]) -> Dict[str, Any]:
    """SpeechT5-style models need an x-vector; everything else needs nothing."""
    if "speecht5" not in handle.repo_id.lower():
        return {}

    raw = (speaker_embedding or "").strip()
    try:
        index = int(raw) if raw else 7306
    except ValueError:
        raise UnsupportedModelError(f"Speaker index must be a number, got {raw!r}.")

    try:
        vector = _load_xvector(index)
    except UnsupportedModelError:
        raise
    except Exception as exc:
        raise UnsupportedModelError(
            f"{handle.repo_id} needs speaker embeddings and they could not be fetched: {exc}"
        ) from exc
    return {"forward_params": {"speaker_embeddings": vector}}


def _load_xvector(index: int):
    """Fetch one CMU-Arctic speaker x-vector, as a (1, 512) tensor.

    The upstream dataset is script-based and `datasets` v3 dropped script
    support ("Dataset scripts are no longer supported"), so pull the archive
    straight off the Hub and read the .npy entries here instead.
    """
    if index in _XVECTOR_CACHE:
        return _XVECTOR_CACHE[index]

    import io
    import zipfile

    import numpy as np

    torch = require("torch")
    hub = require("huggingface_hub")

    archive_path = hub.hf_hub_download(
        repo_id=_XVECTOR_REPO,
        filename=_XVECTOR_ARCHIVE,
        repo_type="dataset",
        token=hf_token(),
    )
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".npy"))
        if not names:
            raise UnsupportedModelError(f"No speaker embeddings found in {_XVECTOR_ARCHIVE}.")
        payload = archive.read(names[index % len(names)])

    vector = torch.tensor(np.load(io.BytesIO(payload))).reshape(1, -1).float()
    _XVECTOR_CACHE[index] = vector
    return vector


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


def _target_sr(pipe) -> int:
    """Sample rate the model's feature extractor expects."""
    extractor = getattr(pipe, "feature_extractor", None)
    return int(getattr(extractor, "sampling_rate", 16000) or 16000)


def _decode_file(path: str, target_sr: int) -> Dict[str, Any]:
    """Decode an audio file into a mono float32 array.

    Handed a filename, the transformers ASR pipeline shells out to the `ffmpeg`
    binary, which is not guaranteed to exist in a Space image (and fails with
    "ffmpeg was not found" when it does not). Decoding in Python via
    soundfile/librosa keeps the whole path dependency-free.
    """
    import numpy as np

    try:
        import soundfile as sf

        array, rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        librosa = require("librosa", "Needed to decode audio files.")
        array, rate = librosa.load(path, sr=None, mono=True)

    return _to_pipeline_input(np.asarray(array, dtype="float32"), int(rate), target_sr)


def _to_pipeline_input(array, rate: int, target_sr: int) -> Dict[str, Any]:
    """Down-mix to mono, scale to [-1, 1] and resample to what the model wants."""
    import numpy as np

    array = np.asarray(array, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)

    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:  # int16 samples, e.g. straight from the microphone
        array = array / 32768.0

    if rate != target_sr:
        librosa = require("librosa", "Needed to resample audio.")
        array = librosa.resample(array, orig_sr=rate, target_sr=target_sr)

    return {"array": array, "sampling_rate": target_sr}


def _as_asr_input(audio_input: Any, target_sr: int = 16000) -> Any:
    """Accept a filepath, or Gradio's (sample_rate, ndarray) tuple."""
    if isinstance(audio_input, str):
        return _decode_file(audio_input, target_sr)
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        rate, array = audio_input
        return _to_pipeline_input(array, int(rate), target_sr)
    return audio_input
