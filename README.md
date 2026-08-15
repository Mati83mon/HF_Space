---
title: Model Playground
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Universal HF playground for text/image/audio/video
---

# ⚡ Model Playground

Universal Hugging Face **Space** for testing any model from the Hub across four
modalities. Paste a `repo_id`, let the app detect the task (or override it), run.

```
Text  ── transformers ── text-generation / chat / classification / QA ...
Image ── diffusers ───── SD · SDXL · FLUX · img2img · inpainting
Audio ── transformers ── TTS · ASR · audio-classification
Video ── diffusers ───── LTX-Video · SVD · CogVideoX · HunyuanVideo · Pyramid-Flow
```

Theme: cyberpunk — black, dark grey, neon yellow, monospaced.

---

## Tabs

| Tab | What it does |
|-----|--------------|
| **Text** | Prompt + sampling params (temperature, top-p, top-k, max tokens, repetition penalty, stop sequences). `Run` for one shot, `Stream` for token-by-token output. |
| **Image** | Prompt / negative prompt, steps, CFG, resolution, seed, batch size. Optional init image + strength for image-to-image and inpainting. |
| **Audio** | TTS (text in, audio out) and ASR (upload or microphone in, transcript out), plus audio classification. |
| **Video** | text-to-video, image-to-video, image-to-gif. Frames, FPS, steps, CFG, resolution, seed, `.mp4`/`.gif` export. |
| **Instruction Playground** | Compose `system` / `user` / `assistant` / `tool` turns, preview the exact string the model's chat template produces, then stream the completion. |
| **Custom API** | Raw JSON kwargs forwarded to the resolved loader; unsupported kwargs are dropped rather than crashing. Also has `Inspect model` for Hub metadata. |
| **Safety & Modes** | Standard vs Uncensored test mode, intended/out-of-scope use, model cache view and `Unload all models`. |

## Example `repo_id`s

```text
text   Qwen/Qwen2.5-1.5B-Instruct · HuggingFaceTB/SmolLM2-360M-Instruct
image  black-forest-labs/FLUX.1-schnell · stabilityai/sdxl-turbo · stabilityai/stable-diffusion-xl-base-1.0
audio  openai/whisper-small · microsoft/speecht5_tts · suno/bark-small
video  Lightricks/LTX-Video · stabilityai/stable-video-diffusion-img2vid-xt · THUDM/CogVideoX-2b
```

## How task detection works

1. **Task override** in the tab (dropdown) — always wins.
2. `pipeline_tag` from `huggingface_hub.model_info`.
3. Tag / repo-name heuristics (`flux`, `svd`, `whisper`, `cogvideo`, …).
4. `library_name == "diffusers"` → `text-to-image`.
5. Otherwise: a readable error telling you to set the override manually.

Detection degrades gracefully when the Hub is unreachable — it falls back to
heuristics instead of failing.

Note that auto-detection follows the Hub, which is not always what you want: e.g.
`Lightricks/LTX-Video` carries `pipeline_tag: image-to-video`, so text-to-video
generation with it needs the override set to `text-to-video`.

## Model cache

Weights live in a process-global LRU cache (`loaders/common.py::MODEL_CACHE`), so
re-running the same model never re-downloads or re-materialises it. Size is capped
by `PLAYGROUND_MAX_CACHED_MODELS` (default `2`); eviction runs `gc.collect()` and
`torch.cuda.empty_cache()`. `Unload all models` in **Safety & Modes** frees everything.

## Safety & Modes

- **Standard** (default) — optional library-side filters (e.g. the Diffusers
  `safety_checker`) stay enabled when a model provides one.
- **Uncensored test mode** — those optional filters are switched off after an
  explicit acknowledgement, so raw model output is shown as-is.

This Space adds **no** filtering of its own in either mode, and never modifies or
circumvents a model's own alignment. Switching modes clears the pipeline cache,
because the setting is baked in at load time. You remain responsible for generated
content and for each model's licence and the Hugging Face Content Policy.

## Optional WebSocket backend

`ws_protocol.py` defines a transport-agnostic JSON frame protocol
(`request` / `cancel` → `ack` / `chunk` / `done` / `error`) served on
`/ws/text`, `/ws/audio`, `/ws/video`.

```bash
# alongside the Space
PLAYGROUND_ENABLE_WS=1 python app.py

# or standalone (this is the path to a Docker Space)
uvicorn ws_protocol:app --host 0.0.0.0 --port 7861
```

## Local development

```bash
pip install -r requirements.txt
python -m py_compile app.py model_registry.py ws_protocol.py loaders/*.py
python app.py            # http://localhost:7860
```

Environment variables: `HF_TOKEN` (gated repos), `PLAYGROUND_MAX_CACHED_MODELS`,
`PLAYGROUND_ENABLE_WS`, `PLAYGROUND_WS_PORT`, `GRADIO_SERVER_NAME/PORT`.

## Deploying as a Space

1. Create a Space → SDK **Gradio**.
2. Push this repository to the Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
3. The YAML block at the top of this README is the Space config (`sdk`, `app_file`,
   `python_version`, …) — keep it as the first thing in the file.
4. Pick hardware in **Settings → Hardware**:

   | Workload | Suggested tier |
   |----------|----------------|
   | small text models, ASR | CPU basic / CPU upgrade |
   | SDXL, FLUX schnell, TTS | T4 small / A10G small |
   | LTX-Video, CogVideoX, HunyuanVideo | A10G large / A100 |

5. Add `HF_TOKEN` in **Settings → Secrets** for gated repos (FLUX dev, Llama, …).

## Repository layout

```
app.py              Gradio UI, tab wiring, mode handling
model_registry.py   task detection + task → modality → loader mapping
ws_protocol.py      WebSocket frame protocol + optional FastAPI app
loaders/
  common.py         model cache, device/dtype, errors, temp files
  text.py           transformers: generation, streaming, chat templates
  image.py          diffusers: text2img / img2img / inpainting
  audio.py          transformers: TTS + ASR
  video.py          diffusers: t2v / i2v / gif export
assets/theme.css    cyberpunk theme (black / grey / neon yellow)
SPEC.md             architecture specification
```

Adding your own loader: implement `load_model(repo_id, task, **opts) -> ModelHandle`
and `run_inference(handle, **params)` in a new module, then register the task in
`model_registry.TASK_MODALITY` and `get_loader()`.

## Limitations

- Video and large diffusion models need GPU hardware; on CPU they are unusably slow.
- Some video repos require a specific `diffusers` version or a custom pipeline class —
  add it to `loaders/video.py::KNOWN_PIPELINES`.
- MP4 export needs `imageio[ffmpeg]`; GIF export is the fallback.
- Gated repos need `HF_TOKEN`.
