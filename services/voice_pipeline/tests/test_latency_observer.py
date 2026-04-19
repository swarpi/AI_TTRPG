"""
tests/test_latency_observer.py
================================
Unit tests for LatencyState, STTTimestampRecorder, and TTSStartRecorder.

These run without any network connections — all Pipecat frame processing
is exercised by calling process_frame() directly with mocked push_frame.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

# Import from parent package (run tests from services/voice_pipeline/)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from latency_observer import LatencyState, STTTimestampRecorder, TTSStartRecorder


# ---------------------------------------------------------------------------
# LatencyState unit tests
# ---------------------------------------------------------------------------

class TestLatencyState:
    @pytest.mark.asyncio
    async def test_initial_state_is_empty(self):
        state = LatencyState()
        assert state.t_stt is None
        assert state.transcript == ""

    @pytest.mark.asyncio
    async def test_record_stt_sets_timestamp_and_transcript(self):
        state = LatencyState()
        await state.record_stt("Hello GM")
        assert state.t_stt is not None
        assert state.transcript == "Hello GM"

    @pytest.mark.asyncio
    async def test_consume_tts_start_returns_and_clears(self):
        state = LatencyState()
        await state.record_stt("Hello GM")
        t, transcript = await state.consume_tts_start()
        assert t is not None
        assert transcript == "Hello GM"
        # State must be cleared
        assert state.t_stt is None
        assert state.transcript == ""

    @pytest.mark.asyncio
    async def test_consume_without_record_returns_none(self):
        state = LatencyState()
        t, transcript = await state.consume_tts_start()
        assert t is None
        assert transcript == ""

    @pytest.mark.asyncio
    async def test_second_consume_returns_none(self):
        """Subsequent TTSAudioRawFrames for same utterance must not re-log."""
        state = LatencyState()
        await state.record_stt("Roll to attack")
        await state.consume_tts_start()  # first — consumes
        t, _ = await state.consume_tts_start()  # second — empty
        assert t is None


# ---------------------------------------------------------------------------
# STTTimestampRecorder unit tests
# ---------------------------------------------------------------------------

class TestSTTTimestampRecorder:
    def _make_processor(self, state: LatencyState) -> STTTimestampRecorder:
        proc = STTTimestampRecorder(state)
        proc.push_frame = AsyncMock()
        return proc

    @pytest.mark.asyncio
    async def test_transcription_frame_stamps_state(self):
        state = LatencyState()
        proc = self._make_processor(state)
        frame = TranscriptionFrame(text="Hello GM", user_id="u1", timestamp="")

        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        assert state.t_stt is not None
        assert state.transcript == "Hello GM"

    @pytest.mark.asyncio
    async def test_empty_transcription_ignored(self):
        state = LatencyState()
        proc = self._make_processor(state)
        frame = TranscriptionFrame(text="   ", user_id="u1", timestamp="")

        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        assert state.t_stt is None

    @pytest.mark.asyncio
    async def test_upstream_transcription_ignored(self):
        """Upstream frames must not trigger timestamp recording."""
        state = LatencyState()
        proc = self._make_processor(state)
        frame = TranscriptionFrame(text="Hello GM", user_id="u1", timestamp="")

        await proc.process_frame(frame, FrameDirection.UPSTREAM)

        assert state.t_stt is None

    @pytest.mark.asyncio
    async def test_non_transcription_frame_passes_through(self):
        state = LatencyState()
        proc = self._make_processor(state)
        # A raw audio frame should pass through without touching state
        audio_frame = MagicMock(spec=AudioRawFrame)
        await proc.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        assert state.t_stt is None


# ---------------------------------------------------------------------------
# TTSStartRecorder unit tests
# ---------------------------------------------------------------------------

class TestTTSStartRecorder:
    def _make_processor(self, state: LatencyState) -> TTSStartRecorder:
        proc = TTSStartRecorder(state)
        proc.push_frame = AsyncMock()
        return proc

    @pytest.mark.asyncio
    async def test_first_audio_frame_logs_latency(self, caplog):
        import logging
        state = LatencyState()
        await state.record_stt("Hello GM")

        proc = self._make_processor(state)
        audio_frame = MagicMock(spec=TTSAudioRawFrame)

        # Should not raise and state should be consumed
        await proc.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        assert state.t_stt is None  # consumed

    @pytest.mark.asyncio
    async def test_second_audio_frame_does_not_consume(self):
        """Only the first audio frame per utterance should trigger logging."""
        state = LatencyState()
        await state.record_stt("Attack the goblin")
        proc = self._make_processor(state)
        audio_frame = MagicMock(spec=TTSAudioRawFrame)

        await proc.process_frame(audio_frame, FrameDirection.DOWNSTREAM)  # 1st: consumed
        await proc.process_frame(audio_frame, FrameDirection.DOWNSTREAM)  # 2nd: no-op

        # state was already consumed on first call
        t, _ = await state.consume_tts_start()
        assert t is None

    @pytest.mark.asyncio
    async def test_audio_frame_without_stt_is_noop(self):
        """If no STT timestamp is set, the recorder must be silent."""
        state = LatencyState()
        proc = self._make_processor(state)
        audio_frame = MagicMock(spec=TTSAudioRawFrame)

        # Should not raise — just pass through
        await proc.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        assert state.t_stt is None
