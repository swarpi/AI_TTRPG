"""
voice_pipeline/echo_processor.py
==================================
Converts Deepgram ``TranscriptionFrame`` into a ``TTSSpeakFrame`` so ElevenLabs
can synthesise it without an LLM in the loop.

This keeps Ticket 2.1 latency measurement clean: STT → echo → TTS with zero
LLM latency, proving the transport and audio services are wired correctly.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class EchoProcessor(FrameProcessor):
    """
    Drop-in pipeline stage that echoes every final STT transcript back through TTS.

    Non-TranscriptionFrames (system frames, audio, etc.) are passed through
    unchanged so the pipeline's control signals keep flowing.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text.strip()
        ):
            text = frame.text.strip()
            logger.debug(f"[ECHO] Speaking: \"{text}\"")
            # Emit the TTS request instead of forwarding the raw transcript
            await self.push_frame(TTSSpeakFrame(text=text), direction)
        else:
            # Pass all other frames (EndFrame, SystemFrame, audio, …) through
            await self.push_frame(frame, direction)
