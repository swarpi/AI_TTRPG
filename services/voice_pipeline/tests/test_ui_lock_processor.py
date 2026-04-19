"""
tests/test_ui_lock_processor.py
===============================
Unit tests for the UILockProcessor.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock
import pytest

from pipecat.frames.frames import OutputTransportMessageFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from semantic_router import RoutedTranscriptionFrame
from ui_lock_processor import UILockProcessor


class TestUILockProcessor:
    @pytest.mark.asyncio
    async def test_emits_ui_lock_if_write_signal_is_true(self):
        proc = UILockProcessor()
        proc.push_frame = AsyncMock()

        frame = RoutedTranscriptionFrame(
            text="I swing my sword",
            user_id="u1",
            timestamp="",
            write_signal=True,
            matched_intent="I swing my sword at the orc"
        )

        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Expected: OutputTransportMessageFrame pushed first, then the original frame
        assert proc.push_frame.call_count == 2
        
        lock_call, orig_call = proc.push_frame.call_args_list
        
        lock_frame = lock_call[0][0]
        assert isinstance(lock_frame, OutputTransportMessageFrame)
        assert lock_frame.message == {"action": "UI_LOCK"}
        assert lock_call[0][1] == FrameDirection.DOWNSTREAM
        
        orig_frame = orig_call[0][0]
        assert orig_frame is frame

    @pytest.mark.asyncio
    async def test_ignores_if_write_signal_is_false(self):
        proc = UILockProcessor()
        proc.push_frame = AsyncMock()

        frame = RoutedTranscriptionFrame(
            text="What color is the sky?",
            user_id="u1",
            timestamp="",
            write_signal=False
        )

        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Expected: Only the original frame is pushed
        proc.push_frame.assert_called_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_ignores_normal_transcription_frames(self):
        proc = UILockProcessor()
        proc.push_frame = AsyncMock()

        frame = TranscriptionFrame(
            text="General chat",
            user_id="u1",
            timestamp=""
        )

        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        proc.push_frame.assert_called_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_ignores_upstream_frames(self):
        proc = UILockProcessor()
        proc.push_frame = AsyncMock()

        frame = RoutedTranscriptionFrame(
            text="I swing my sword",
            user_id="u1",
            timestamp="",
            write_signal=True
        )

        await proc.process_frame(frame, FrameDirection.UPSTREAM)

        # Expected: Should hit the pass-through exactly once
        proc.push_frame.assert_called_once_with(frame, FrameDirection.UPSTREAM)

