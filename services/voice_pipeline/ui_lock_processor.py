"""
ui_lock_processor.py
====================
Ticket 3.1 — WebRTC Data Channel UI_LOCK Broadcast

Intercepts RoutedTranscriptionFrames. If the router flagged the frame with
`write_signal: True`, this processor emits a JSON control payload down the
WebRTC Data Channel requesting the client UI to "lock" (e.g., show a spinner
or grey out inputs) while the backend processes the game action.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import Frame, OutputTransportMessageFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from semantic_router import RoutedTranscriptionFrame


class UILockProcessor(FrameProcessor):
    """
    Emits a {"action": "UI_LOCK"} message down the Data Channel whenever
    a RoutedTranscriptionFrame with write_signal=True is detected.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # We only care about downstream RoutedTranscriptionFrames
        if isinstance(frame, RoutedTranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.write_signal:
                logger.debug(f"[UI_LOCK] Emitting lock for intent: '{frame.matched_intent}'")
                # 1. Instantly emit the Data Channel message
                lock_frame = OutputTransportMessageFrame(message={"action": "UI_LOCK"})
                await self.push_frame(lock_frame, direction)

        # 2. Always pass the original frame through
        await self.push_frame(frame, direction)
