# SPEC — Universal Model Playground (HuggingFace Space)

Specyfikacja architektury projektu. Dokument powstał w **FAZIE 1 (PLAN)** i jest
źródłem prawdy dla implementacji (FAZA 2) oraz weryfikacji (FAZA 3).

---

## 1. Cel

Jeden Space (SDK: `gradio`), który pozwala załadować **dowolny** model z Hugging Face
Hub po `repo_id` i uruchomić na nim inferencję w czterech modalnościach:

| Modalność | Przykładowe zadania | Biblioteka |
|-----------|--------------------|------------|
| Text  | `text-generation`, `text2text-generation`, `text-classification`, chat / instruction-following | `transformers` |
| Image | `text-to-image`, `image-to-image` (SD, SDXL, FLUX, ControlNet-ready) | `diffusers` |
| Audio | `text-to-speech`, `automatic-speech-recognition`, `audio-classification` | `transformers` |
| Video | `text-to-video`, `image-to-video`, `image-to-gif` (LTX-Video, SVD, CogVideoX, HunyuanVideo, Pyramid-Flow miniFLUX, FLUX 3) | `diffusers` |

Dodatkowo: zakładka **Custom API** (surowe kwargs JSON), panel **Safety & Modes**
(Standard vs Uncensored test mode) oraz **Instruction Playground** (system/user/assistant/tool).

---

## 2. Struktura repozytorium

```
HF_Space/
├── app.py                 # Gradio UI: Text | Image | Audio | Video | Instruction | Custom API | Safety
├── model_registry.py      # wykrywanie zadania + mapowanie task -> modalność -> loader
├── ws_protocol.py         # opisany protokół WebSocket (/ws/text, /ws/audio, /ws/video) + opcjonalny FastAPI
├── loaders/
│   ├── __init__.py        # re-eksport loaderów
│   ├── common.py          # cache wag, device/dtype, wyjątki, helpery plikowe
│   ├── text.py            # load_model() / run_inference() / stream_inference()
│   ├── image.py           # load_model() / run_inference()
│   ├── audio.py           # load_model() / run_inference()  (TTS + ASR)
│   └── video.py           # load_model() / run_inference()  (t2v / i2v / gif)
├── assets/
│   └── theme.css          # motyw cyberpunk: czerń / ciemna szarość / neonowa żółć
├── requirements.txt
├── README.md              # + blok YAML konfiguracji Space
├── SPEC.md                # ten dokument
└── .gitignore
```

### Odpowiedzialność modułów

- **`app.py`** — wyłącznie warstwa prezentacji i sklejenie zdarzeń Gradio z loaderami.
  Żadnej logiki ML poza wywołaniami `loaders.*`. Buduje `gr.Blocks`, wstrzykuje CSS,
  trzyma stan trybu (Standard / Uncensored) w `gr.State`.
- **`model_registry.py`** — jedyne miejsce, które „wie”, jaki task obsługuje jaki loader.
  Odpytuje Hub (`huggingface_hub.model_info`) o `pipeline_tag`, `tags`, `library_name`,
  wnioskuje modalność, pozwala ręcznie nadpisać. Zawiera listę przykładowych `repo_id`.
- **`loaders/common.py`** — wspólna infrastruktura: globalny cache modeli (`MODEL_CACHE`),
  detekcja `device`/`dtype`, `UnsupportedModelError`, zapisywanie wyników do plików tymczasowych.
- **`loaders/*.py`** — czyste funkcje `load_model(repo_id, task, **opts)` i
  `run_inference(handle, **params)`. Loadery **nie** importują Gradio.
- **`ws_protocol.py`** — deklaratywny opis ramek JSON dla przyszłego Docker Space
  (FastAPI + WebSocket). Import `fastapi` jest opcjonalny — brak paczki nie psuje Space.

---

## 3. Ładowanie modeli z Hub

### 3.1 Wykrywanie typu zadania

```
repo_id ─► huggingface_hub.model_info(repo_id)
             ├─ pipeline_tag   (najsilniejszy sygnał)
             ├─ library_name   ("diffusers" / "transformers" / "timm" ...)
             └─ tags           (fallback: "text-to-video", "flux", "lora", ...)
                     │
                     ▼
        TASK_MODALITY[task] ─► "text" | "image" | "audio" | "video"
                     │
                     ▼
              loaders.<modality>
```

Reguły rozstrzygania (kolejność):
1. **Ręczny wybór użytkownika** (dropdown `Task override`) — zawsze wygrywa.
2. `pipeline_tag`, o ile jest w `TASK_MODALITY`.
3. Heurystyki po `tags` / nazwie repo (`flux`, `stable-diffusion`, `video`, `svd`, `tts`).
4. `library_name == "diffusers"` → domyślnie `text-to-image`.
5. Brak dopasowania → `UnsupportedModelError` z czytelnym komunikatem w UI
   (zamiast traceback'a) i podpowiedzią „ustaw Task override ręcznie”.

Detekcja jest w pełni offline-tolerant: gdy Hub jest nieosiągalny, zwracany jest
rekord z `source="heuristic"` zamiast wyjątku.

### 3.2 Cache wag (brak przeładowań)

Wagi trzymane są w **module-level dict** `loaders.common.MODEL_CACHE`,
kluczowanym `(modality, repo_id, task, dtype, extra)`. Kolejne wywołania
`load_model()` z tymi samymi parametrami zwracają ten sam obiekt pipeline.
Cache ma limit (`MAX_CACHED_MODELS`, domyślnie 2 — GPU Spaces mają mało VRAM)
i politykę LRU z `gc.collect()` + `torch.cuda.empty_cache()` przy eksmisji.
UI wystawia przycisk **Unload all models**.

### 3.3 Uchwyt modelu (`ModelHandle`)

`load_model()` zwraca `ModelHandle` — lekki dataclass:
`repo_id`, `task`, `modality`, `pipeline` (obiekt), `meta` (dict).
Dzięki temu `run_inference()` jest czystą funkcją `handle × params -> wynik`.

---

## 4. Safety & Modes + Instruction Playground

### 4.1 Panel Safety & Modes

Dwa tryby przechowywane w `gr.State`:

| Tryb | Zachowanie |
|------|-----------|
| **Standard** (domyślny) | `safety_checker` w pipeline'ach Diffusers pozostaje włączony, jeżeli model go dostarcza. Widoczna nota o przeznaczeniu. |
| **Uncensored test mode** | Space **nie dokłada** własnego filtrowania i wyłącza `safety_checker`/`requires_safety_checker` w Diffusers. Wymaga zaznaczenia checkboxa potwierdzającego; UI pokazuje trwały baner ostrzegawczy. |

Kluczowe: Space nigdy nie modyfikuje ani nie obchodzi wyrównania samego modelu —
przełącznik dotyczy wyłącznie opcjonalnych filtrów post-hoc dostarczanych przez
bibliotekę. Panel zawiera sekcję **Intended use / Out-of-scope use** w duchu
Model Cards, plus przypomnienie o odpowiedzialności użytkownika za wygenerowane treści.
Zmiana trybu unieważnia cache pipeline'ów obrazu/wideo (inna konfiguracja).

### 4.2 Instruction Playground

Zakładka do budowania struktur promptów dla modeli językowych:
- edytowalne pola `system`, `user`, `assistant` (few-shot prefix), `tool` (JSON),
- render przez `tokenizer.apply_chat_template()` gdy model ma szablon czatu,
  w przeciwnym razie prosty fallback `ROLE: treść`,
- podgląd **dokładnego** stringu wysyłanego do modelu (pole „Rendered prompt”),
- uruchomienie z tymi samymi parametrami samplingu co zakładka Text.

---

## 5. Streaming / real-time

- **Tekst w Gradio**: funkcje generatorowe (`yield`) + `TextIteratorStreamer`
  z `transformers` w osobnym wątku. Przycisk **Stream** korzysta z generatora,
  **Run** ze zwykłego wywołania.
- **Obraz/Wideo**: progres przez `gr.Progress` i callback `callback_on_step_end`
  z Diffusers (podgląd numeru kroku).
- **WebSocket** (`ws_protocol.py`): zdefiniowany protokół ramek JSON na endpointach
  `/ws/text`, `/ws/audio`, `/ws/video`:

```jsonc
// client -> server
{"type": "request", "id": "…", "modality": "text", "repo_id": "…", "task": "…", "params": {…}}
{"type": "cancel",  "id": "…"}
// server -> client
{"type": "ack",    "id": "…"}
{"type": "chunk",  "id": "…", "seq": 0, "data": "…", "encoding": "text|base64"}
{"type": "done",   "id": "…", "meta": {…}}
{"type": "error",  "id": "…", "message": "…"}
```

Ten sam dispatcher (`ws_protocol.handle_request`) obsługuje wszystkie modalności,
więc migracja do Docker Space = uruchomienie `uvicorn ws_protocol:build_app()`.

---

## 6. Motyw UI (cyberpunk: żółty / szary / czarny)

`assets/theme.css` definiuje zmienne CSS i nadpisuje tokeny Gradio:

| Token | Wartość |
|-------|---------|
| tło główne | `#050505` |
| panele | `#1E1E1E` |
| obramowania / linie | `#3C3C3C` |
| akcent (neon) | `#FFD700` |
| akcent wtórny | `#FFA800` |
| ostrzeżenie (uncensored) | `#FF4B4B` |
| font | `JetBrains Mono`, `IBM Plex Mono`, `monospace` |

Podpięcie dwutorowe: `gr.themes.Base(...)` ustawia paletę na poziomie
tokenów Gradio, a `css=` (wczytany `assets/theme.css`) dokłada glow, animacje
hover na `Run` / `Stream`, scanline w nagłówku i styl panelu logów.
Brak pliku CSS nie wywraca aplikacji (fallback na pusty string).

---

## 7. Obsługa błędów

- Każdy handler UI opakowany w `safe_call`, który zwraca `(wynik, log)` zamiast rzucać.
- `UnsupportedModelError` → czytelny komunikat + sugestia Task override.
- Brak zainstalowanej biblioteki (`diffusers`, `torch`) → komunikat „zainstaluj X /
  wybierz hardware GPU”, a nie `ImportError` na starcie. Wszystkie ciężkie importy
  są **leniwe** (wewnątrz funkcji), dzięki czemu `python -m py_compile app.py`
  i sam start UI działają bez pełnego stosu ML.

---

## 8. Kryteria akceptacji (FAZA 3 – VERIFY)

1. `python -m py_compile app.py model_registry.py ws_protocol.py loaders/*.py` — bez błędów.
2. Import `app.py` nie wymaga `torch` / `diffusers` / `transformers`.
3. `model_registry.resolve()` poprawnie mapuje przykładowe repo_id na modalności
   (również bez dostępu do sieci — ścieżka heurystyczna).
4. `demo` (obiekt `gr.Blocks`) buduje się bez wyjątku.
5. README zawiera poprawny blok YAML konfiguracji Space.
