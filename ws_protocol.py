"""Streaming protocol for the optional WebSocket backend.

The Gradio Space itself streams through generator functions; this module defines
the wire format and a modality-agnostic dispatcher so the same playground can be
re-deployed as a Docker Space (FastAPI + uvicorn) without touching the loaders.

Wire format (JSON frames)
-------------------------
client -> server
    {"type": "request", "id": "<uuid>", "modality": "text|image|audio|video",
     "repo_id": "...", "task": "auto|<task>", "params": {...}}
    {"type": "cancel",  "id": "<uuid>"}

server -> client
    {"type": "ack",   "id": "..."}
    {"type": "chunk", "id": "...", "seq": 0, "data": "...", "encoding": "text|base64"}
    {"type": "done",  "id": "...", "meta": {...}}
    {"type": "error", "id": "...", "message": "..."}

Endpoints: /ws/text, /ws/audio, /ws/video (one dispatcher, three routes).

Run standalone:  uvicorn ws_protocol:app --host 0.0.0.0 --port 7861
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, Optional

import model_registry
from loaders.common import PlaygroundError

WS_ROUTES = ("/ws/text", "/ws/audio", "/ws/video")

FRAME_TYPES = ("request", "cancel", "ack", "chunk", "done", "error")


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


@dataclass
class Frame:
    type: str
    id: str
    seq: Optional[int] = None
    data: Optional[str] = None
    encoding: Optional[str] = None
    message: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if v not in (None, {}, [])}
        payload["type"] = self.type
        payload["id"] = self.id
        return json.dumps(payload, ensure_ascii=False)


def ack(request_id: str) -> Frame:
    return Frame(type="ack", id=request_id)


def chunk(request_id: str, seq: int, data: str, encoding: str = "text") -> Frame:
    return Frame(type="chunk", id=request_id, seq=seq, data=data, encoding=encoding)


def done(request_id: str, **meta: Any) -> Frame:
    return Frame(type="done", id=request_id, meta=meta)


def error(request_id: str, message: str) -> Frame:
    return Frame(type="error", id=request_id, message=message)


def parse_request(raw: str) -> Dict[str, Any]:
    """Validate an inbound frame and return it as a dict."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlaygroundError(f"Malformed frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlaygroundError("Frame must be a JSON object.")
    frame_type = payload.get("type")
    if frame_type not in FRAME_TYPES:
        raise PlaygroundError(f"Unknown frame type {frame_type!r}.")
    if frame_type == "request":
        for required in ("id", "repo_id"):
            if not payload.get(required):
                raise PlaygroundError(f"Frame is missing '{required}'.")
    return payload


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def handle_request(payload: Dict[str, Any]) -> Iterator[Frame]:
    """Turn one `request` frame into a stream of response frames.

    Synchronous generator on purpose: the transport (FastAPI, Gradio, a queue
    worker) decides how to pump it, and the loaders stay transport-agnostic.
    """
    request_id = payload.get("id", "0")
    repo_id = payload.get("repo_id", "")
    task = payload.get("task", "auto")
    params: Dict[str, Any] = payload.get("params", {}) or {}

    yield ack(request_id)

    try:
        spec = model_registry.resolve(repo_id, task)
        loader = model_registry.get_loader(spec.modality)
        handle = loader.load_model(
            spec.repo_id,
            spec.task,
            dtype=params.get("dtype", "auto"),
            uncensored=bool(params.get("uncensored", False)),
        )

        if spec.modality == "text":
            prompt = params.pop("prompt", "")
            seq = 0
            for partial in loader.stream_inference(handle, prompt, **params):
                yield chunk(request_id, seq, partial)
                seq += 1
            yield done(request_id, task=spec.task, chunks=seq)
            return

        if spec.modality == "image":
            images, info = loader.run_inference(handle, **params)
            for seq, image in enumerate(images):
                yield chunk(request_id, seq, _encode_image(image), encoding="base64")
            yield done(request_id, task=spec.task, info=info)
            return

        if spec.modality == "audio":
            audio, text = loader.run_inference(handle, **params)
            if text:
                yield chunk(request_id, 0, text)
            if audio is not None:
                rate, array = audio
                yield chunk(request_id, 1, _encode_array(array), encoding="base64")
                yield done(request_id, task=spec.task, sampling_rate=rate)
                return
            yield done(request_id, task=spec.task)
            return

        path, info = loader.run_inference(handle, **params)
        if path:
            yield chunk(request_id, 0, _encode_file(path), encoding="base64")
        yield done(request_id, task=spec.task, info=info, filename=os.path.basename(path or ""))

    except PlaygroundError as exc:
        yield error(request_id, str(exc))
    except Exception as exc:  # never let the socket die on a model-specific bug
        yield error(request_id, f"{type(exc).__name__}: {exc}")


def _encode_file(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _encode_image(image: Any) -> str:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_array(array: Any) -> str:
    try:
        import numpy as np

        return base64.b64encode(np.asarray(array, dtype="float32").tobytes()).decode("ascii")
    except ImportError:
        return base64.b64encode(bytes(array)).decode("ascii")


# --------------------------------------------------------------------------- #
# Optional FastAPI app (only built when fastapi is installed)
# --------------------------------------------------------------------------- #


def build_app():
    """Return a FastAPI app exposing WS_ROUTES. Raises if fastapi is missing."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    api = FastAPI(title="Model Playground WS")

    async def endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = parse_request(raw)
                except PlaygroundError as exc:
                    await websocket.send_text(error("0", str(exc)).to_json())
                    continue
                if payload["type"] != "request":
                    continue
                for frame in handle_request(payload):
                    await websocket.send_text(frame.to_json())
        except WebSocketDisconnect:
            return

    for route in WS_ROUTES:
        api.add_api_websocket_route(route, endpoint)

    @api.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "routes": list(WS_ROUTES)}

    return api


try:  # convenience for `uvicorn ws_protocol:app`
    app = build_app()
except Exception:  # fastapi absent in a plain Gradio Space — expected
    app = None
