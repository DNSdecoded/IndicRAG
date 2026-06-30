# IndicRAG v2.1 — Audio Transcription for Voice Queries

**Date:** 2026-06-23
**Status:** Draft
**Scope:** Voice query input via browser microphone with multi-provider transcription

## Overview

Add audio input to IndicRAG so users can speak their query instead of typing. Audio is recorded in the browser, sent to a transcription provider, and the resulting text is placed in the query input for review before submission. This extends the existing text-based RAG pipeline without modifying it — transcription is a preprocessing step that produces text.

## Requirements

| Requirement | Detail |
|-------------|--------|
| Use case | Voice queries only (not audio ingestion into the knowledge base) |
| Providers | Sarvam AI, Google Speech-to-Text, OpenAI Whisper API, local faster-whisper |
| Provider selection | User chooses via frontend dropdown; "auto" option available |
| Audio input | Browser microphone via MediaRecorder API |
| Max duration | 2 minutes |
| Audio format | WebM/Opus (browser-native), server also accepts WAV, MP3, OGG, M4A |
| Max file size | 10 MB |
| Review flow | Show-then-confirm (default): transcribed text fills the input box for editing. Quick mode (toggle): sends directly to RAG pipeline |

## Architecture

### Data Flow

```
[Mic Button] → MediaRecorder (WebM/Opus) → POST /transcribe (audio + provider + lang_hint)
    → transcription.py (provider dispatch) → TranscriptionResult
    → Response to frontend → Fill text input (show-then-confirm)
                            OR send to /query directly (quick mode)
```

The existing pipeline (`/query`, `/chat`, `/agent/query`) is unchanged. Transcription is fully decoupled — it produces text, and the text enters the normal flow.

### New Files

| File | Purpose |
|------|---------|
| `transcription.py` | Provider interface, 4 provider implementations, auto-selection logic |

### Modified Files

| File | Change |
|------|--------|
| `config.py` | Add transcription-related config variables |
| `api_server.py` | Add `POST /transcribe` endpoint |
| `requirements.txt` | Add `faster-whisper`, `openai`, `google-cloud-speech` |
| `static/index.html` | Add mic button, recording UI, provider dropdown, quick-mode toggle |
| `.env.example` | Add transcription API key placeholders |

## Detailed Design

### 1. `transcription.py` — Provider Module

Follows the project's one-module-per-concern pattern (like `translation.py`, `rerank.py`).

#### Data Types

```python
@dataclass
class TranscriptionResult:
    text: str               # transcribed text
    language: str           # detected/confirmed language code (ISO 639-1)
    confidence: float       # 0.0-1.0, provider-dependent
    provider: str           # provider name used
    duration_ms: int        # audio duration processed
```

#### Provider Protocol

```python
class TranscriptionProvider(Protocol):
    name: str

    def transcribe(
        self,
        audio_bytes: bytes,
        format: str,
        lang_hint: str | None = None,
    ) -> TranscriptionResult: ...

    def is_available(self) -> bool: ...
```

#### Provider Implementations

**SarvamProvider:**
- Uses Sarvam AI's ASR API via HTTP (`httpx`).
- Requires `SARVAM_API_KEY`.
- Strongest Indic language support natively. Preferred by auto-mode for Indic languages.

**GoogleSTTProvider:**
- Uses Google Cloud Speech-to-Text v2 API.
- Requires `GOOGLE_STT_API_KEY` (API key) or `GOOGLE_APPLICATION_CREDENTIALS` (service account JSON path). API key is simpler; service account is more secure for production.
- Good multilingual support, tight integration with existing Google ecosystem.

**WhisperAPIProvider:**
- Uses OpenAI's Whisper API.
- Requires `OPENAI_API_KEY`.
- Strong general-purpose ASR, good Indic support via large model.

**LocalWhisperProvider:**
- Uses `faster-whisper` library (CTranslate2-optimized Whisper).
- No API key required. Model downloaded to `models/` cache dir on first use.
- Model size configurable via `LOCAL_WHISPER_MODEL` (default: `"small"`).
- Runs on CPU or GPU (auto-detected, same as existing torch device logic).
- Preferred by auto-mode for English and when minimizing cost.

#### Singleton & Thread Safety

Follows the same lazy-loading pattern as `translation.py`:
- Global `_providers: dict[str, TranscriptionProvider]` initialized to `{}`.
- `_lock = threading.Lock()` guards initialization.
- `get_provider(name: str) -> TranscriptionProvider` loads on first call with double-checked locking.
- Local Whisper model loaded once and kept in memory (same lifecycle as BGE-M3, NLLB).

#### Auto-Selection Logic

When provider is `"auto"`:
1. If `lang_hint` is an Indic language and Sarvam is available → use Sarvam.
2. Else if local Whisper is available → use local Whisper (zero cost).
3. Else → use the first available cloud provider (Google STT > Whisper API).
4. If nothing is available → raise an error listing which API keys to set.

#### Public API

```python
def transcribe(
    audio_bytes: bytes,
    format: str = "webm",
    provider: str = "auto",
    lang_hint: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text using the specified provider."""
```

This is the only function other modules call.

### 2. API Endpoint — `POST /transcribe`

Added to `api_server.py`.

**Request:** `multipart/form-data`
- `audio` (file, required): Audio file. Max 10 MB.
- `provider` (string, optional): `"sarvam"`, `"google"`, `"whisper"`, `"local"`, or `"auto"`. Default: value of `TRANSCRIPTION_PROVIDER_DEFAULT`.
- `lang_hint` (string, optional): ISO 639-1 language code hint (e.g., `"hi"`, `"ta"`, `"en"`).

**Response:** `TranscribeResponse`
```json
{
  "text": "What are the effects of climate change on Indian agriculture?",
  "language": "en",
  "confidence": 0.92,
  "provider": "local",
  "duration_ms": 3200
}
```

**Validation:**
- File size > 10 MB → `413 Payload Too Large`
- Unsupported format → `415 Unsupported Media Type` (accepted: webm, wav, mp3, ogg, m4a)
- Unknown provider → `400 Bad Request`
- Provider unavailable (no API key / model not loaded) → `503 Service Unavailable` with message specifying which env var to set
- Empty transcription result → `422 Unprocessable Entity`

**Auth:** Uses existing `verify_api_key` dependency (same as `/query`).

### 3. Frontend — Mic Button & Recording UI

All changes in `static/index.html` (the single-page app).

#### Mic Button
- Positioned next to the send button in the chat input area.
- Icon: microphone SVG. States: idle (default), recording (red/pulsing), processing (spinner).
- Click to start recording, click again to stop. Auto-stops at 2 minutes.

#### Recording State UI
- While recording: pulsing red dot + elapsed timer (e.g., "0:12") displayed near the input area.
- While processing: spinner replaces the mic button, "Transcribing..." text.

#### Transcription Flow (Default — Show-Then-Confirm)
1. User clicks mic → recording starts.
2. User clicks mic again (or 2 min auto-stop) → recording stops.
3. Audio sent to `POST /transcribe` with selected provider.
4. Transcribed text fills the text input box. User can edit.
5. User clicks send (or presses Enter) to submit the query normally.

#### Transcription Flow (Quick Mode)
1. Steps 1-3 same as above.
2. Transcribed text is immediately sent to `/query` or `/chat` (depending on current mode).
3. Transcribed text shown in the chat as the user's message.

#### Provider & Settings
- Provider dropdown in the settings/sidebar area: Auto, Sarvam, Google, Whisper, Local.
- Quick-mode toggle next to the provider dropdown.
- Both persist in `localStorage`.

#### Browser Compatibility
- `MediaRecorder` API with `audio/webm;codecs=opus` — supported in Chrome, Firefox, Edge, Safari 14.1+.
- If `MediaRecorder` is not available, mic button is hidden (graceful degradation).

### 4. Configuration — `config.py`

```python
# Transcription Providers
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GOOGLE_STT_API_KEY = os.getenv("GOOGLE_STT_API_KEY", "")  # or use GOOGLE_APPLICATION_CREDENTIALS for service account
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Transcription Settings
TRANSCRIPTION_PROVIDER_DEFAULT = os.getenv("TRANSCRIPTION_PROVIDER_DEFAULT", "auto")
TRANSCRIPTION_MAX_DURATION_SEC = int(os.getenv("TRANSCRIPTION_MAX_DURATION_SEC", "120"))
TRANSCRIPTION_MAX_FILE_SIZE_MB = int(os.getenv("TRANSCRIPTION_MAX_FILE_SIZE_MB", "10"))
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "small")
```

### 5. Dependencies — `requirements.txt`

```
# Audio transcription
faster-whisper>=1.1.0          # Local Whisper (CTranslate2)
openai>=1.40.0                 # OpenAI Whisper API
google-cloud-speech>=2.27.0    # Google Speech-to-Text v2
httpx>=0.27.0                  # HTTP client for Sarvam API
```

Note: `httpx` is used for Sarvam because they may not have an official Python SDK. If they do, prefer the SDK.

### 6. Error Handling

| Scenario | Behavior |
|----------|----------|
| No providers available (no keys, no local model) | `/transcribe` returns 503 listing which env vars to set |
| Audio too long (> 2 min) | Client-side auto-stop; server-side 413 if file > 10 MB |
| Transcription confidence < 0.3 | Response includes `confidence` field; frontend shows a warning icon next to the transcribed text |
| Empty transcription (silence / noise) | 422 error; frontend shows "Could not detect speech. Please try again." |
| Provider API error (rate limit, timeout) | 502 with provider-specific error message |
| Unsupported browser (no MediaRecorder) | Mic button hidden; text input works as usual |

### 7. Testing

**Unit tests** (no API keys needed):
- Provider auto-selection logic (mock `is_available()`)
- Format validation (accepted vs rejected formats)
- File size validation
- `TranscriptionResult` construction
- Each provider's `transcribe()` method with mocked HTTP responses / model

**Integration tests** (`@pytest.mark.integration`):
- Real transcription with each cloud provider (requires API keys)
- Local Whisper transcription with a short test audio file

**Frontend:**
- Manual testing: mic recording flow, provider switching, quick-mode toggle, show-then-confirm flow

## Out of Scope

- Audio file ingestion (transcribing lectures/talks into the vector store)
- Real-time streaming transcription (WebSocket-based)
- Speaker diarization
- Audio-to-audio (speech in, speech out)
- Custom fine-tuned ASR models

## API Key Summary

| Variable | Provider | Required? |
|----------|----------|-----------|
| `SARVAM_API_KEY` | Sarvam AI ASR | Optional — needed only if using Sarvam provider |
| `GOOGLE_STT_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` | Google Speech-to-Text | Optional — API key or service account path; needed only if using Google provider |
| `OPENAI_API_KEY` | OpenAI Whisper API | Optional — needed only if using Whisper API provider |
| (none) | Local faster-whisper | No key needed — model auto-downloaded to `models/` |

At least one provider must be available for transcription to work. The local model requires no credentials and is the recommended default for cost-conscious deployments.
