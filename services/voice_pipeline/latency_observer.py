"""
voice_pipeline/latency_observer.py
====================================
Measures STT-output → TTS-first-audio-chunk latency without touching pipeline logic.

Architecture
------------
Two lightweight FrameProcessors share a ``LatencyState`` object:

  transport.input()
    → DeepgramSTT          ← emits TranscriptionFrame
    → STTTimestampRecorder ← records t_stt = perf_counter()
    → EchoProcessor        ← TranscriptionFrame → TTSSpeakFrame
    → ElevenLabsTTS        ← emits TTSAudioRawFrame
    → TTSStartRecorder     ← on first TTSAudioRawFrame: log latency, clear state
    → transport.output()

Thread safety: ``LatencyState`` is protected by an asyncio.Lock because
Pipecat may run frames concurrently on the event loop.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class LatencyState:
    """Thread-safe container shared between the two timestamp processors."""

    t_stt: Optional[float] = field(default=None)
    transcript: str = field(default="")
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def record_stt(self, transcript: str) -> None:
        """Called when Deepgram emits a final TranscriptionFrame."""
        async with self._lock:
            self.t_stt = time.perf_counter()
            self.transcript = transcript
            logger.debug(f"[LATENCY] t_stt stamped for: \"{transcript}\"")

    async def consume_tts_start(self) -> tuple[Optional[float], str]:
        """
        Called on first TTSAudioRawFrame per utterance.
        Returns (t_stt, transcript) and clears state so subsequent audio
        frames for the same utterance don't trigger a second log line.
        """
        async with self._lock:
            t = self.t_stt
            transcript = self.transcript
            self.t_stt = None       # reset — subsequent audio frames are ignored
            self.transcript = ""
            return t, transcript


# ---------------------------------------------------------------------------
# Processor 1 — sits just after STT
# ---------------------------------------------------------------------------

class STTTimestampRecorder(FrameProcessor):
    """
    Records the wall-clock time when a final Deepgram TranscriptionFrame arrives.
    Passes all frames through unmodified.
    """

    def __init__(self, state: LatencyState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text.strip()
        ):
            await self._state.record_stt(frame.text.strip())


# ---------------------------------------------------------------------------
# Processor 2 — sits just after TTS
# ---------------------------------------------------------------------------

class TTSStartRecorder(FrameProcessor):
    """
    On the first TTSAudioRawFrame for each utterance, computes and logs
    the STT-output → TTS-first-audio-chunk round-trip latency.

    Log format:
        [LATENCY] stt→tts_start 312.4 ms | transcript="Hello GM"
    """

    def __init__(self, state: LatencyState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, TTSAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            t_stt, transcript = await self._state.consume_tts_start()
            if t_stt is not None:
                latency_ms = (time.perf_counter() - t_stt) * 1000
                logger.info(
                    f'[LATENCY] stt→tts_start {latency_ms:.1f} ms'
                    f' | transcript="{transcript}"'
                )
