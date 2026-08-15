"""Universal Model Playground — a Gradio Space for testing any Hugging Face model.

Tabs: Text | Image | Audio | Video | Instruction Playground | Custom API | Safety & Modes.

This module is UI only: every ML call goes through `model_registry` + `loaders.*`,
and every heavy import happens inside those modules, lazily.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr

import model_registry
from loaders.common import PlaygroundError, cache_summary, clear_cache, get_device

APP_DIR = Path(__file__).parent
CSS_PATH = APP_DIR / "assets" / "theme.css"


# --------------------------------------------------------------------------- #
# ZeroGPU support
# --------------------------------------------------------------------------- #

try:
    import spaces  # pre-installed on ZeroGPU Spaces

    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False


# ZeroGPU caps how long a single GPU reservation may last, and adds its own
# overhead on top of the requested duration. Asking for more than the account's
# ceiling fails the call outright ("requested GPU duration is larger than the
# maximum allowed") before any work starts, so the default stays conservative;
# raise it via the env var on a tier that allows longer slots.
MAX_GPU_DURATION = int(os.environ.get("PLAYGROUND_GPU_DURATION", "120"))


def gpu_task(duration: int = 120):
    """Reserve a ZeroGPU slot for the wrapped handler.

    On ZeroGPU hardware a GPU only exists for the duration of a `spaces.GPU`
    call, so every handler that touches a model needs this. The decorator is a
    no-op both off ZeroGPU (package absent) and on classic CPU/GPU tiers (the
    package detects it), which keeps a single code path for all hardware.
    """

    def decorator(fn):
        if not _HAS_SPACES:
            return fn
        return spaces.GPU(duration=min(duration, MAX_GPU_DURATION))(fn)

    return decorator


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


def load_css() -> str:
    try:
        return CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""  # a missing stylesheet must never take the Space down


CYBER_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#050505",
    body_text_color="#e8e8e8",
    background_fill_primary="#1e1e1e",
    background_fill_secondary="#141414",
    border_color_primary="#3c3c3c",
    block_title_text_color="#ffd700",
    button_primary_background_fill="#ffd700",
    button_primary_text_color="#0a0a0a",
    button_secondary_background_fill="#141414",
    button_secondary_text_color="#ffd700",
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _fmt_error(exc: BaseException) -> str:
    if isinstance(exc, PlaygroundError):
        return f"[error] {exc}"
    return f"[error] {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


def _prepare(repo_id: str, task_override: str, expected_modality: str, uncensored: bool, dtype: str = "auto"):
    """Resolve a repo_id, check it belongs on this tab, and load (or reuse) it."""
    spec = model_registry.resolve(repo_id, task_override)
    if spec.modality != expected_modality:
        raise PlaygroundError(
            f"'{spec.repo_id}' looks like a {spec.modality} model (task={spec.task}), "
            f"but you are on the {expected_modality.capitalize()} tab. "
            f"Use the {spec.modality.capitalize()} tab, or set a Task override."
        )
    loader = model_registry.get_loader(spec.modality)
    handle = loader.load_model(spec.repo_id, spec.task, dtype=dtype, uncensored=uncensored)
    return loader, handle, spec


def _log(spec, handle, extra: str = "") -> str:
    parts = [spec.summary(), f"handle   : {handle.describe()}", f"device   : {get_device()}"]
    if extra:
        parts.append(extra)
    parts.append(f"\n[cache]\n{cache_summary()}")
    return "\n".join(parts)


def _task_choices(modality: str) -> List[str]:
    return ["auto"] + model_registry.TASKS_BY_MODALITY[modality]


# --------------------------------------------------------------------------- #
# Text tab handlers
# --------------------------------------------------------------------------- #


def _sampling(temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw) -> Dict[str, Any]:
    from loaders.text import parse_stop_sequences

    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "stop_sequences": parse_stop_sequences(stop_raw),
    }


@gpu_task(duration=60)
def text_run(repo_id, task, prompt, temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw, uncensored):
    try:
        loader, handle, spec = _prepare(repo_id, task, "text", uncensored)
        params = _sampling(temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw)
        output = loader.run_inference(handle, prompt, **params)
        return output, _log(spec, handle)
    except BaseException as exc:
        return "", _fmt_error(exc)


@gpu_task(duration=60)
def text_stream(repo_id, task, prompt, temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw, uncensored):
    try:
        loader, handle, spec = _prepare(repo_id, task, "text", uncensored)
        params = _sampling(temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw)
        log = _log(spec, handle, "mode     : streaming")
    except BaseException as exc:
        yield "", _fmt_error(exc)
        return

    try:
        for partial in loader.stream_inference(handle, prompt, **params):
            yield partial, log
    except BaseException as exc:
        yield "", _fmt_error(exc)


# --------------------------------------------------------------------------- #
# Instruction playground
# --------------------------------------------------------------------------- #


def _build_messages(system: str, user: str, assistant_prefix: str, tool_json: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    if tool_json.strip():
        # Tool definitions differ per template; passing them as a `tool` message
        # keeps the transcript honest without pretending to be a real tool API.
        try:
            parsed = json.loads(tool_json)
            content = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError as exc:
            raise PlaygroundError(f"Tool JSON is invalid: {exc}")
        messages.append({"role": "tool", "content": content})
    if user.strip():
        messages.append({"role": "user", "content": user.strip()})
    if assistant_prefix.strip():
        messages.append({"role": "assistant", "content": assistant_prefix.strip()})
    if not messages:
        raise PlaygroundError("Nothing to send — fill in at least the user message.")
    return messages


@gpu_task(duration=60)
def instruction_render(repo_id, task, system, user, assistant_prefix, tool_json, uncensored):
    try:
        loader, handle, spec = _prepare(repo_id, task, "text", uncensored)
        messages = _build_messages(system, user, assistant_prefix, tool_json)
        rendered = loader.render_chat(handle, messages)
        has_template = bool(getattr(getattr(handle.pipeline, "tokenizer", None), "chat_template", None))
        note = "chat template: model-provided" if has_template else "chat template: generic fallback"
        return rendered, _log(spec, handle, note)
    except BaseException as exc:
        return "", _fmt_error(exc)


@gpu_task(duration=60)
def instruction_run(
    repo_id, task, system, user, assistant_prefix, tool_json,
    temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw, uncensored,
):
    try:
        loader, handle, spec = _prepare(repo_id, task, "text", uncensored)
        messages = _build_messages(system, user, assistant_prefix, tool_json)
        rendered = loader.render_chat(handle, messages)
        params = _sampling(temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop_raw)
        log = _log(spec, handle, "mode     : instruction playground")
    except BaseException as exc:
        yield "", "", _fmt_error(exc)
        return

    try:
        for partial in loader.stream_inference(handle, rendered, **params):
            yield rendered, partial, log
    except BaseException as exc:
        yield rendered, "", _fmt_error(exc)


# --------------------------------------------------------------------------- #
# Image / Audio / Video handlers
# --------------------------------------------------------------------------- #


@gpu_task(duration=90)
def image_run(
    repo_id, task, prompt, negative_prompt, steps, guidance, width, height,
    seed, num_images, init_image, strength, uncensored,
):
    try:
        loader, handle, spec = _prepare(repo_id, task, "image", uncensored)
        images, info = loader.run_inference(
            handle,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            seed=int(seed),
            num_images=int(num_images),
            init_image=init_image,
            strength=float(strength),
        )
        return images, _log(spec, handle, f"run      : {info}")
    except BaseException as exc:
        return None, _fmt_error(exc)


@gpu_task(duration=60)
def audio_run(repo_id, task, text, audio_input, speaker, language, timestamps, uncensored):
    try:
        loader, handle, spec = _prepare(repo_id, task, "audio", uncensored)
        audio, text_out = loader.run_inference(
            handle,
            text=text,
            audio_input=audio_input,
            speaker_embedding=speaker or None,
            language=language or None,
            return_timestamps=bool(timestamps),
        )
        return audio, text_out, _log(spec, handle)
    except BaseException as exc:
        return None, "", _fmt_error(exc)


@gpu_task(duration=120)
def video_run(
    repo_id, task, prompt, negative_prompt, init_image, num_frames, fps,
    steps, guidance, width, height, seed, output_format, uncensored,
):
    try:
        loader, handle, spec = _prepare(repo_id, task, "video", uncensored)
        path, info = loader.run_inference(
            handle,
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            num_frames=int(num_frames),
            fps=int(fps),
            steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            seed=int(seed),
            output_format=output_format,
        )
        video = path if path and path.endswith(".mp4") else None
        return video, path, _log(spec, handle, f"run      : {info}")
    except BaseException as exc:
        return None, None, _fmt_error(exc)


# --------------------------------------------------------------------------- #
# Custom API tab
# --------------------------------------------------------------------------- #


@gpu_task(duration=90)
def custom_run(repo_id, task, params_json, uncensored):
    """Send raw kwargs straight to whichever loader the repo resolves to."""
    try:
        params = json.loads(params_json) if params_json.strip() else {}
        if not isinstance(params, dict):
            raise PlaygroundError("Params must be a JSON object.")
    except json.JSONDecodeError as exc:
        return "", None, None, None, f"[error] Invalid JSON: {exc}"

    try:
        spec = model_registry.resolve(repo_id, task)
        loader = model_registry.get_loader(spec.modality)
        handle = loader.load_model(
            spec.repo_id, spec.task,
            dtype=params.pop("dtype", "auto"),
            uncensored=uncensored,
        )

        if spec.modality == "text":
            prompt = params.pop("prompt", "")
            return loader.run_inference(handle, prompt, **params), None, None, None, _log(spec, handle)

        if spec.modality == "image":
            images, info = loader.run_inference(handle, **params)
            return info, images, None, None, _log(spec, handle)

        if spec.modality == "audio":
            audio, text_out = loader.run_inference(handle, **params)
            return text_out, None, audio, None, _log(spec, handle)

        path, info = loader.run_inference(handle, **params)
        video = path if path and path.endswith(".mp4") else None
        return info, None, None, video, _log(spec, handle)
    except BaseException as exc:
        return "", None, None, None, _fmt_error(exc)


def inspect_model(repo_id, task):
    try:
        spec = model_registry.resolve(repo_id, task)
        info = model_registry.fetch_model_info(spec.repo_id)
        return spec.summary() + "\n\n[hub metadata]\n" + json.dumps(info, indent=2, default=str)
    except BaseException as exc:
        return _fmt_error(exc)


# --------------------------------------------------------------------------- #
# Safety & Modes
# --------------------------------------------------------------------------- #

STANDARD_NOTE = """<div class="pg-note">
<h3>Standard mode</h3>
Optional post-hoc filters shipped by the libraries (e.g. the Diffusers
<code>safety_checker</code>) stay enabled when the model provides one.
This is the default for shared or public deployments.
</div>"""

UNCENSORED_NOTE = """<div class="pg-warning">
<strong>UNCENSORED TEST MODE ACTIVE.</strong><br>
Optional library-side filters are switched off, so raw model output is shown as-is.
This playground adds no filtering of its own in either mode — it never modifies or
circumvents a model's own alignment. Output may be inaccurate, offensive or unsafe;
you are responsible for what you generate, publish and distribute, and for complying
with each model's licence and the Hugging Face Content Policy.
</div>"""


def set_mode(mode: str, acknowledged: bool):
    """Flip the mode. Uncensored requires an explicit acknowledgement tick."""
    if mode == "Uncensored test mode":
        if not acknowledged:
            return False, STANDARD_NOTE + (
                '<div class="pg-warning">Tick the acknowledgement box to enable '
                "uncensored test mode. Staying in Standard mode.</div>"
            ), "Standard"
        # Filter settings are baked into diffusion pipelines at load time.
        clear_cache()
        return True, UNCENSORED_NOTE, mode
    clear_cache()
    return False, STANDARD_NOTE, "Standard"


def unload_all():
    return clear_cache() + "\n\n[cache]\n" + cache_summary()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

HEADER = """<div id="pg-header">
  <h1>&gt;&gt; Model Playground</h1>
  <p>Universal Hugging Face inference bench — text · image · audio · video ·
  custom API. Paste a <code>repo_id</code>, pick a task, run.</p>
</div>"""


def _model_bar(modality: str, default_repo: str):
    """repo_id + task override + inspect button, shared layout across tabs."""
    with gr.Row():
        repo = gr.Textbox(
            label="repo_id",
            value=default_repo,
            placeholder="owner/model",
            scale=3,
        )
        task = gr.Dropdown(
            label="Task override",
            choices=_task_choices(modality),
            value="auto",
            scale=2,
        )
    gr.Examples(
        examples=[[m] for m in model_registry.EXAMPLE_MODELS[modality]],
        inputs=[repo],
        label=f"Example {modality} models",
    )
    return repo, task


def _sampling_controls():
    with gr.Row():
        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = off)")
    with gr.Row():
        max_new_tokens = gr.Slider(16, 4096, value=256, step=16, label="Max new tokens")
        repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, step=0.01, label="Repetition penalty")
        stop = gr.Textbox(label="Stop sequences (comma separated)", value="")
    return temperature, top_p, top_k, max_new_tokens, repetition_penalty, stop


def build_demo() -> gr.Blocks:
    with gr.Blocks(theme=CYBER_THEME, css=load_css(), title="Model Playground") as demo:
        uncensored_state = gr.State(False)
        gr.HTML(HEADER)

        # ------------------------------- TEXT ------------------------------ #
        with gr.Tabs():
            with gr.Tab("Text"):
                t_repo, t_task = _model_bar("text", "Qwen/Qwen2.5-1.5B-Instruct")
                t_prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Write a haiku about neon rain...")
                (t_temp, t_top_p, t_top_k, t_max, t_rep, t_stop) = _sampling_controls()
                with gr.Row():
                    t_run = gr.Button("Run", variant="primary", elem_classes="pg-run")
                    t_stream = gr.Button("Stream", variant="secondary")
                t_out = gr.Textbox(label="Output", lines=12, elem_classes="pg-output", show_copy_button=True)
                t_log = gr.Textbox(label="Log", lines=8, elem_classes="pg-log")

                t_inputs = [t_repo, t_task, t_prompt, t_temp, t_top_p, t_top_k, t_max, t_rep, t_stop, uncensored_state]
                t_run.click(text_run, inputs=t_inputs, outputs=[t_out, t_log])
                t_stream.click(text_stream, inputs=t_inputs, outputs=[t_out, t_log])

            # ------------------------------ IMAGE ---------------------------- #
            with gr.Tab("Image"):
                i_repo, i_task = _model_bar("image", "stabilityai/sdxl-turbo")
                i_prompt = gr.Textbox(label="Prompt", lines=3, value="a rain-soaked neon alley, cyberpunk, 35mm")
                i_negative = gr.Textbox(label="Negative prompt", lines=2, value="")
                with gr.Row():
                    i_steps = gr.Slider(1, 100, value=25, step=1, label="Steps")
                    i_cfg = gr.Slider(0.0, 20.0, value=7.0, step=0.1, label="Guidance (CFG)")
                    i_num = gr.Slider(1, 4, value=1, step=1, label="Images")
                with gr.Row():
                    i_width = gr.Slider(256, 1536, value=768, step=64, label="Width")
                    i_height = gr.Slider(256, 1536, value=768, step=64, label="Height")
                    i_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                with gr.Accordion("Image-to-image / inpainting input", open=False):
                    i_init = gr.Image(label="Init image", type="pil")
                    i_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Strength")
                i_run = gr.Button("Run", variant="primary", elem_classes="pg-run")
                i_gallery = gr.Gallery(label="Output", columns=2, height=520)
                i_log = gr.Textbox(label="Log", lines=8, elem_classes="pg-log")

                i_run.click(
                    image_run,
                    inputs=[i_repo, i_task, i_prompt, i_negative, i_steps, i_cfg, i_width, i_height,
                            i_seed, i_num, i_init, i_strength, uncensored_state],
                    outputs=[i_gallery, i_log],
                )

            # ------------------------------ AUDIO ---------------------------- #
            with gr.Tab("Audio"):
                a_repo, a_task = _model_bar("audio", "openai/whisper-small")
                a_text = gr.Textbox(label="Text (TTS input)", lines=3, value="Systems online. Welcome to the grid.")
                a_audio = gr.Audio(label="Audio (ASR input)", sources=["upload", "microphone"], type="filepath")
                with gr.Row():
                    a_speaker = gr.Textbox(label="Speaker index (SpeechT5 x-vector)", value="")
                    a_language = gr.Textbox(label="Language hint (ASR)", value="")
                    a_timestamps = gr.Checkbox(label="Return timestamps", value=False)
                a_run = gr.Button("Run", variant="primary", elem_classes="pg-run")
                a_out_audio = gr.Audio(label="Generated audio")
                a_out_text = gr.Textbox(label="Text output", lines=6, elem_classes="pg-output")
                a_log = gr.Textbox(label="Log", lines=8, elem_classes="pg-log")

                a_run.click(
                    audio_run,
                    inputs=[a_repo, a_task, a_text, a_audio, a_speaker, a_language, a_timestamps, uncensored_state],
                    outputs=[a_out_audio, a_out_text, a_log],
                )

            # ------------------------------ VIDEO ---------------------------- #
            with gr.Tab("Video"):
                v_repo, v_task = _model_bar("video", "Lightricks/LTX-Video")
                v_prompt = gr.Textbox(label="Prompt", lines=3, value="a drone shot over a neon megacity at night")
                v_negative = gr.Textbox(label="Negative prompt", lines=2, value="worst quality, blurry, jittery")
                v_init = gr.Image(label="Init image (image-to-video / image-to-gif)", type="pil")
                with gr.Row():
                    v_frames = gr.Slider(8, 257, value=49, step=1, label="Frames")
                    v_fps = gr.Slider(1, 30, value=8, step=1, label="FPS")
                    v_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                with gr.Row():
                    v_cfg = gr.Slider(0.0, 20.0, value=6.0, step=0.1, label="Guidance (CFG)")
                    v_width = gr.Slider(256, 1280, value=704, step=32, label="Width")
                    v_height = gr.Slider(256, 1280, value=480, step=32, label="Height")
                with gr.Row():
                    v_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                    v_format = gr.Radio(["mp4", "gif"], value="mp4", label="Output format")
                v_run = gr.Button("Run", variant="primary", elem_classes="pg-run")
                v_out = gr.Video(label="Output video")
                v_file = gr.File(label="Output file (mp4 / gif)")
                v_log = gr.Textbox(label="Log", lines=8, elem_classes="pg-log")

                v_run.click(
                    video_run,
                    inputs=[v_repo, v_task, v_prompt, v_negative, v_init, v_frames, v_fps, v_steps,
                            v_cfg, v_width, v_height, v_seed, v_format, uncensored_state],
                    outputs=[v_out, v_file, v_log],
                )

            # ------------------------- INSTRUCTION ---------------------------- #
            with gr.Tab("Instruction Playground"):
                gr.HTML(
                    '<div class="pg-note"><h3>Instruction call playground</h3>'
                    "Compose system / user / assistant / tool turns, inspect the exact string "
                    "produced by the model's own chat template, then run it.</div>"
                )
                i2_repo, i2_task = _model_bar("text", "Qwen/Qwen2.5-1.5B-Instruct")
                ip_system = gr.Textbox(label="System", lines=4, value="You are a terse, precise assistant.")
                ip_user = gr.Textbox(label="User", lines=5, value="Explain diffusion models in three bullet points.")
                ip_assistant = gr.Textbox(label="Assistant prefix (optional)", lines=2, value="")
                ip_tools = gr.Textbox(
                    label="Tool / function definitions (JSON, optional)",
                    lines=5,
                    value="",
                    placeholder='{"name": "get_weather", "parameters": {"city": "string"}}',
                )
                (ip_temp, ip_top_p, ip_top_k, ip_max, ip_rep, ip_stop) = _sampling_controls()
                with gr.Row():
                    ip_render = gr.Button("Render prompt", variant="secondary")
                    ip_run = gr.Button("Run", variant="primary", elem_classes="pg-run")
                ip_rendered = gr.Textbox(label="Rendered prompt", lines=10, elem_classes="pg-log", show_copy_button=True)
                ip_out = gr.Textbox(label="Output", lines=10, elem_classes="pg-output", show_copy_button=True)
                ip_log = gr.Textbox(label="Log", lines=6, elem_classes="pg-log")

                ip_render.click(
                    instruction_render,
                    inputs=[i2_repo, i2_task, ip_system, ip_user, ip_assistant, ip_tools, uncensored_state],
                    outputs=[ip_rendered, ip_log],
                )
                ip_run.click(
                    instruction_run,
                    inputs=[i2_repo, i2_task, ip_system, ip_user, ip_assistant, ip_tools,
                            ip_temp, ip_top_p, ip_top_k, ip_max, ip_rep, ip_stop, uncensored_state],
                    outputs=[ip_rendered, ip_out, ip_log],
                )

            # ------------------------- CUSTOM API ----------------------------- #
            with gr.Tab("Custom API"):
                gr.HTML(
                    '<div class="pg-note"><h3>Custom API</h3>'
                    "Raw keyword arguments are forwarded to the resolved loader. Anything the "
                    "underlying pipeline does not accept is dropped rather than crashing.</div>"
                )
                with gr.Row():
                    c_repo = gr.Textbox(label="repo_id", value="Qwen/Qwen2.5-1.5B-Instruct", scale=3)
                    c_task = gr.Dropdown(
                        label="Task override",
                        choices=["auto"] + sorted(model_registry.TASK_MODALITY),
                        value="auto",
                        scale=2,
                    )
                c_params = gr.Code(
                    label="Params (JSON)",
                    language="json",
                    value='{\n  "prompt": "hello there",\n  "max_new_tokens": 128,\n  "temperature": 0.7\n}',
                )
                with gr.Row():
                    c_inspect = gr.Button("Inspect model", variant="secondary")
                    c_run = gr.Button("Run", variant="primary", elem_classes="pg-run")
                c_text = gr.Textbox(label="Text / info output", lines=10, elem_classes="pg-output")
                c_gallery = gr.Gallery(label="Image output", columns=2, height=360)
                c_audio = gr.Audio(label="Audio output")
                c_video = gr.Video(label="Video output")
                c_log = gr.Textbox(label="Log", lines=10, elem_classes="pg-log")

                c_run.click(
                    custom_run,
                    inputs=[c_repo, c_task, c_params, uncensored_state],
                    outputs=[c_text, c_gallery, c_audio, c_video, c_log],
                )
                c_inspect.click(inspect_model, inputs=[c_repo, c_task], outputs=[c_log])

            # ------------------------ SAFETY & MODES -------------------------- #
            with gr.Tab("Safety & Modes"):
                s_mode = gr.Radio(
                    ["Standard", "Uncensored test mode"],
                    value="Standard",
                    label="Mode",
                )
                s_ack = gr.Checkbox(
                    label="I understand this disables optional library-side filters and I accept "
                          "responsibility for the generated content.",
                    value=False,
                )
                s_banner = gr.HTML(STANDARD_NOTE)
                gr.HTML(
                    '<div class="pg-note"><h3>Intended use</h3>'
                    "Hands-on evaluation of Hub models across modalities: prompt sensitivity, "
                    "instruction following, sampling parameters, latency and output quality."
                    "<h4>Out-of-scope use</h4>"
                    "Production serving, unattended public deployment, generating content that is "
                    "illegal, targets real people, or violates a model's licence or the Hugging Face "
                    "Content Policy. This Space performs no content moderation in either mode."
                    "</div>"
                )
                with gr.Row():
                    s_unload = gr.Button("Unload all models", elem_classes="pg-danger")
                    s_refresh = gr.Button("Refresh cache view", variant="secondary")
                s_cache = gr.Textbox(label="Model cache", lines=8, value=cache_summary(), elem_classes="pg-log")

                s_mode.change(set_mode, inputs=[s_mode, s_ack], outputs=[uncensored_state, s_banner, s_mode])
                s_ack.change(set_mode, inputs=[s_mode, s_ack], outputs=[uncensored_state, s_banner, s_mode])
                s_unload.click(unload_all, outputs=[s_cache])
                s_refresh.click(lambda: cache_summary(), outputs=[s_cache])

        gr.HTML(
            '<div class="pg-note">Weights are cached per process — re-running a model does not '
            "re-download it. See <code>SPEC.md</code> for architecture and "
            "<code>ws_protocol.py</code> for the optional WebSocket streaming backend.</div>"
        )

    return demo


def _maybe_start_ws_server() -> None:
    """Optionally run the WebSocket backend alongside Gradio (opt-in via env)."""
    if os.environ.get("PLAYGROUND_ENABLE_WS") != "1":
        return
    try:
        import threading

        import uvicorn

        import ws_protocol

        api = ws_protocol.build_app()
        port = int(os.environ.get("PLAYGROUND_WS_PORT", "7861"))
        thread = threading.Thread(
            target=uvicorn.run,
            args=(api,),
            kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"},
            daemon=True,
        )
        thread.start()
        print(f"[playground] WebSocket backend on :{port} {ws_protocol.WS_ROUTES}")
    except Exception as exc:
        print(f"[playground] WebSocket backend not started: {exc}")


demo = build_demo()

if __name__ == "__main__":
    _maybe_start_ws_server()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_api=True,
        # Gradio 5's experimental SSR layer sits in front of the app and rejects
        # the POSTs this UI makes ("405 Method Not Allowed"), so every Run button
        # failed before reaching Python. Keep it off.
        ssr_mode=False,
        # Surface exceptions raised outside our handlers (e.g. GPU scheduling)
        # instead of a bare "Error" toast with an empty log box.
        show_error=True,
    )
