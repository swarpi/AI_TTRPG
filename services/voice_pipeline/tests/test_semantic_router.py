"""
tests/test_semantic_router.py
================================
Unit tests for SemanticRouter and RoutedTranscriptionFrame.

ALL tests run 100% offline — no Redis, no API keys, no network calls.
The AsyncSearchIndex and the encoder function are both mocked.

Test matrix
-----------
1.  "I swing my sword"       → write_signal: True  (GAME_ACTION)
2.  "What color is the sky?" → write_signal: False (GENERAL)
3.  Non-TranscriptionFrame   → passed through unchanged
4.  Empty transcript         → passed through unchanged
5.  Upstream frame           → passed through unchanged
6.  Redis failure            → write_signal: False  (safe default)
7.  Empty Redis results      → write_signal: False  (safe default)
8.  RoutedTranscriptionFrame → not re-routed (idempotent)
9.  Frame schema             → all required fields present, correct types
10. _classify static method  → direct unit tests of classification logic
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from semantic_router import RoutedTranscriptionFrame, SemanticRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DUMMY_EMBEDDING = np.zeros(384, dtype=np.float32)


def _make_encoder(embedding: np.ndarray = DUMMY_EMBEDDING):
    """Return a synchronous encoder that always returns the given embedding."""
    def encoder(text: str) -> np.ndarray:
        return embedding
    return encoder


def _make_index(results: list[dict] | Exception) -> AsyncMock:
    """Return a mocked AsyncSearchIndex.query() that returns the given results."""
    index = MagicMock()
    if isinstance(results, Exception):
        index.query = AsyncMock(side_effect=results)
    else:
        index.query = AsyncMock(return_value=results)
    return index


def _make_processor(results: list[dict] | Exception = None) -> SemanticRouter:
    """Convenience: build a SemanticRouter with mocked index and encoder."""
    index = _make_index(results or [])
    proc  = SemanticRouter(index=index, encoder=_make_encoder())
    proc.push_frame = AsyncMock()
    return proc


def _transcript(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u1", timestamp="")


# ---------------------------------------------------------------------------
# 1 & 2. Core routing — GAME_ACTION vs GENERAL
# ---------------------------------------------------------------------------

class TestCoreRouting:
    @pytest.mark.asyncio
    async def test_game_action_gives_write_signal_true(self):
        """'I swing my sword' → nearest is GAME_ACTION → write_signal=True."""
        results = [{"label": "GAME_ACTION", "utterance": "I swing my sword at the orc", "vector_distance": "0.08"}]
        proc = _make_processor(results)

        await proc.process_frame(_transcript("I swing my sword"), FrameDirection.DOWNSTREAM)

        proc.push_frame.assert_called_once()
        pushed: RoutedTranscriptionFrame = proc.push_frame.call_args[0][0]
        assert isinstance(pushed, RoutedTranscriptionFrame)
        assert pushed.write_signal is True
        assert pushed.matched_label == "GAME_ACTION"
        assert "swing" in pushed.matched_intent.lower() or "sword" in pushed.matched_intent.lower()

    @pytest.mark.asyncio
    async def test_off_topic_gives_write_signal_false(self):
        """'What color is the sky?' → nearest is GENERAL → write_signal=False."""
        results = [{"label": "GENERAL", "utterance": "What color is the sky?", "vector_distance": "0.05"}]
        proc = _make_processor(results)

        await proc.process_frame(_transcript("What color is the sky?"), FrameDirection.DOWNSTREAM)

        proc.push_frame.assert_called_once()
        pushed: RoutedTranscriptionFrame = proc.push_frame.call_args[0][0]
        assert pushed.write_signal is False
        assert pushed.matched_label == "GENERAL"

    @pytest.mark.asyncio
    async def test_text_is_preserved_in_routed_frame(self):
        results = [{"label": "GAME_ACTION", "utterance": "I attack", "vector_distance": "0.10"}]
        proc = _make_processor(results)

        await proc.process_frame(_transcript("I attack the dragon"), FrameDirection.DOWNSTREAM)

        pushed: RoutedTranscriptionFrame = proc.push_frame.call_args[0][0]
        assert pushed.text == "I attack the dragon"

    @pytest.mark.asyncio
    async def test_similarity_score_is_computed(self):
        """score = 1 - cosine_distance; distance 0.08 → score ~ 0.92."""
        results = [{"label": "GAME_ACTION", "utterance": "I attack", "vector_distance": "0.08"}]
        proc = _make_processor(results)

        await proc.process_frame(_transcript("stab the goblin"), FrameDirection.DOWNSTREAM)

        pushed: RoutedTranscriptionFrame = proc.push_frame.call_args[0][0]
        assert abs(pushed.similarity_score - 0.92) < 1e-6


# ---------------------------------------------------------------------------
# 3–5. Passthrough cases — non-matching frames
# ---------------------------------------------------------------------------

class TestPassthrough:
    @pytest.mark.asyncio
    async def test_non_transcription_frame_passes_through(self):
        proc = _make_processor()
        audio = MagicMock(spec=AudioRawFrame)
        await proc.process_frame(audio, FrameDirection.DOWNSTREAM)

        proc.push_frame.assert_called_once_with(audio, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_empty_transcript_passes_through(self):
        proc = _make_processor()
        await proc.process_frame(_transcript("   "), FrameDirection.DOWNSTREAM)

        proc.push_frame.assert_called_once()
        pushed = proc.push_frame.call_args[0][0]
        assert isinstance(pushed, TranscriptionFrame)
        assert not isinstance(pushed, RoutedTranscriptionFrame)

    @pytest.mark.asyncio
    async def test_upstream_frame_passes_through(self):
        proc = _make_processor()
        frame = _transcript("I attack the goblin")
        await proc.process_frame(frame, FrameDirection.UPSTREAM)

        proc.push_frame.assert_called_once_with(frame, FrameDirection.UPSTREAM)
        # index.query must NOT have been called
        proc._index.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_routed_frame_is_not_re_routed(self):
        """RoutedTranscriptionFrame should pass through unchanged (idempotent)."""
        proc = _make_processor()
        routed = RoutedTranscriptionFrame(
            text="I attack",
            user_id="u1",
            timestamp="",
            write_signal=True,
            matched_label="GAME_ACTION",
        )
        await proc.process_frame(routed, FrameDirection.DOWNSTREAM)

        proc.push_frame.assert_called_once_with(routed, FrameDirection.DOWNSTREAM)
        proc._index.query.assert_not_called()


# ---------------------------------------------------------------------------
# 6–7. Error / empty result handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_redis_error_defaults_to_false(self):
        """If Redis query raises, write_signal must be False (safe default)."""
        proc = _make_processor(ConnectionError("Redis unavailable"))
        await proc.process_frame(_transcript("I attack the orc"), FrameDirection.DOWNSTREAM)

        pushed: RoutedTranscriptionFrame = proc.push_frame.call_args[0][0]
        assert pushed.write_signal is False
        assert pushed.matched_label == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_empty_redis_results_defaults_to_false(self):
        proc = _make_processor([])
        await proc.process_frame(_transcript("I cast a spell"), FrameDirection.DOWNSTREAM)

        pushed: RoutedTranscriptionFrame = proc.push_frame.call_args[0][0]
        assert pushed.write_signal is False
        assert pushed.matched_label == "UNKNOWN"
        assert pushed.similarity_score == 0.0


# ---------------------------------------------------------------------------
# 8. Encoder is run in executor (non-blocking)
# ---------------------------------------------------------------------------

class TestNonBlocking:
    @pytest.mark.asyncio
    async def test_encoder_called_with_correct_text(self):
        """Ensures text is passed to the encoder (i.e., encode was called)."""
        captured: list[str] = []

        def tracking_encoder(text: str) -> np.ndarray:
            captured.append(text)
            return DUMMY_EMBEDDING

        index = _make_index([{"label": "GENERAL", "utterance": "...", "vector_distance": "0.5"}])
        proc = SemanticRouter(index=index, encoder=tracking_encoder)
        proc.push_frame = AsyncMock()

        await proc.process_frame(_transcript("Hello GM"), FrameDirection.DOWNSTREAM)

        assert captured == ["Hello GM"]


# ---------------------------------------------------------------------------
# 9. Schema — RoutedTranscriptionFrame field contract
# ---------------------------------------------------------------------------

class TestRoutedTranscriptionFrameSchema:
    def test_is_transcription_frame_subclass(self):
        """Downstream code using isinstance(frame, TranscriptionFrame) must still work."""
        f = RoutedTranscriptionFrame(text="hello", user_id="u1", timestamp="")
        assert isinstance(f, TranscriptionFrame)

    def test_default_write_signal_is_false(self):
        f = RoutedTranscriptionFrame(text="hello", user_id="u1", timestamp="")
        assert f.write_signal is False

    def test_all_required_fields_present(self):
        f = RoutedTranscriptionFrame(
            text="I attack",
            user_id="u1",
            timestamp="t0",
            write_signal=True,
            similarity_score=0.9,
            matched_label="GAME_ACTION",
            matched_intent="I attack the goblin",
        )
        assert f.text == "I attack"
        assert f.write_signal is True
        assert isinstance(f.similarity_score, float)
        assert f.matched_label == "GAME_ACTION"

    def test_similarity_score_type(self):
        f = RoutedTranscriptionFrame(text="x", user_id="u", timestamp="", similarity_score=0.75)
        assert isinstance(f.similarity_score, float)


# ---------------------------------------------------------------------------
# 10. _classify static method
# ---------------------------------------------------------------------------

class TestClassify:
    def test_game_action_result(self):
        results = [{"label": "GAME_ACTION", "utterance": "I swing my sword", "vector_distance": "0.1"}]
        ws, label, intent, score = SemanticRouter._classify(results)
        assert ws is True
        assert label == "GAME_ACTION"
        assert intent == "I swing my sword"
        assert abs(score - 0.9) < 1e-6

    def test_general_result(self):
        results = [{"label": "GENERAL", "utterance": "What color is the sky?", "vector_distance": "0.05"}]
        ws, label, intent, score = SemanticRouter._classify(results)
        assert ws is False
        assert label == "GENERAL"
        assert abs(score - 0.95) < 1e-6

    def test_empty_results(self):
        ws, label, intent, score = SemanticRouter._classify([])
        assert ws is False
        assert label == "UNKNOWN"
        assert score == 0.0

    def test_missing_distance_defaults(self):
        results = [{"label": "GAME_ACTION", "utterance": "attack"}]   # no vector_distance key
        ws, label, intent, score = SemanticRouter._classify(results)
        assert ws is True
        assert score == 0.0   # 1 - 1.0 fallback distance
