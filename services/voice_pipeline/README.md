# voice_pipeline

**Sprint 2 — Ticket 2.1**: Pipecat WebRTC voice loop.

Ultra-low-latency echo pipeline: browser mic → Deepgram Nova-3 STT → ElevenLabs TTS → browser speaker. Measures and logs the STT-output → TTS-first-audio-chunk latency for every utterance.

---

## Pipeline

```
Browser mic
  └─► Daily WebRTC in
        └─► Deepgram Nova-3 (STT)
              └─► STTTimestampRecorder   ← stamps t_stt
                    └─► EchoProcessor   ← TranscriptionFrame → TTSSpeakFrame
                          └─► ElevenLabs (TTS)
                                └─► TTSStartRecorder   ← logs [LATENCY] N ms
                                      └─► Daily WebRTC out
                                            └─► Browser speaker
```

---

## Setup

### 1. API keys you need

| Service | Sign up | Key name |
|---|---|---|
| [Daily](https://dashboard.daily.co) | Free | `DAILY_API_KEY` |
| [Deepgram](https://console.deepgram.com) | Free $200 | `DEEPGRAM_API_KEY` |
| [ElevenLabs](https://elevenlabs.io) | Free tier | `ELEVENLABS_API_KEY` |

### 2. Install

```bash
cd services/voice_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and fill in the three API keys
```

### 4. Run

```bash
python bot.py
```

You'll see:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎲  Room URL : https://yourname.daily.co/xxxxxx
   Open this URL in your browser and click 'Join'
   Speak something → hear it echoed back via TTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Open the URL, allow microphone, speak — your words come back through ElevenLabs.

---

## Latency Log Format

Every utterance produces a `[LATENCY]` log line:

```
2025-04-16 21:53:12 | INFO | latency_observer:process_frame:72 -
  [LATENCY] stt→tts_start 312.4 ms | transcript="Hello GM"
```

| Field | Meaning |
|---|---|
| `stt→tts_start` | Time from Deepgram `TranscriptionFrame` to first `TTSAudioRawFrame` |
| `N ms` | Wall-clock milliseconds |
| `transcript` | The text Deepgram recognised |

**Target**: `< 600 ms` over a reasonable internet connection.

---

## Tests

```bash
pytest tests/ -v
```

Tests are fully offline — no API keys required.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DAILY_API_KEY` | ✅ | — | Creates ephemeral rooms |
| `DAILY_ROOM_URL` | ❌ | auto | Skip room creation, use this URL |
| `DEEPGRAM_API_KEY` | ✅ | — | Nova-3 streaming STT |
| `ELEVENLABS_API_KEY` | ✅ | — | TTS synthesis |
| `ELEVENLABS_VOICE_ID` | ❌ | `21m00Tcm4TlvDq8ikWAM` | Rachel voice |
| `LOG_LEVEL` | ❌ | `INFO` | `DEBUG` for frame-level tracing |
