"""
semantic_router.py
==================
Ticket 2.2 — L0 Semantic Router Frame Processor

Intercepts downstream TranscriptionFrames, runs a non-blocking RedisVL
vector similarity search against 50 pre-seeded D&D intents, and emits a
RoutedTranscriptionFrame with a write_signal flag.

Pipeline position
-----------------
    STTTimestampRecorder
      → SemanticRouter        ← THIS MODULE
        → EchoProcessor / future LLM processor

Design notes
------------
* The embedding model (fastembed ONNX) is CPU-bound — it runs in a
  ThreadPoolExecutor so the asyncio event loop is NEVER blocked.
* The RedisVL query uses AsyncSearchIndex — purely async, no blocking I/O.
* RoutedTranscriptionFrame is a dataclass subclass of TranscriptionFrame,
  so downstream processors that check isinstance(frame, TranscriptionFrame)
  continue to work without changes (EchoProcessor, latency observer).
* k=1 nearest-neighbour determines write_signal; the matched label is logged
  with the distance for auditability.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# ---------------------------------------------------------------------------
# Custom frame type
# ---------------------------------------------------------------------------

@dataclass
class RoutedTranscriptionFrame(TranscriptionFrame):
    """
    TranscriptionFrame enriched with semantic routing metadata.

    Downstream processors should check write_signal to decide whether
    to invoke the MCP tools (game state mutation) or just respond verbally.
    """
    write_signal: bool = False
    similarity_score: float = 0.0      # 1 - cosine_distance (higher = more similar)
    matched_label: str = "UNKNOWN"     # "GAME_ACTION" | "GENERAL" | "UNKNOWN"
    matched_intent: str = ""           # The nearest-neighbour utterance from the seed set


# ---------------------------------------------------------------------------
# Semantic Router processor
# ---------------------------------------------------------------------------

# Number of nearest neighbours to retrieve (k=1 for POC — nearest wins)
_K = 1


class SemanticRouter(FrameProcessor):
    """
    Async semantic router that tags transcription frames with a write_signal.

    Parameters
    ----------
    index : AsyncSearchIndex
        Initialised RedisVL async index (already connected to Redis).
        Must contain the vectors seeded by seed_redis.py.
    encoder : Callable[[str], np.ndarray]
        Synchronous function that encodes a string to a float32 embedding.
        Called inside run_in_executor to avoid blocking the event loop.
        Example: ``lambda text: list(model.embed([text]))[0]``
    executor : ThreadPoolExecutor, optional
        Shared executor for encoder calls.  A private single-thread pool
        is created if not provided.
    """

    def __init__(
        self,
        index,  # redisvl.index.AsyncSearchIndex
        encoder: Callable[[str], np.ndarray],
        executor: Optional[concurrent.futures.Executor] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._index   = index
        self._encoder = encoder
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="semantic_router",
        )

    # ── core passthrough ──────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, TranscriptionFrame)
            and not isinstance(frame, RoutedTranscriptionFrame)  # don't re-route
            and direction == FrameDirection.DOWNSTREAM
            and frame.text.strip()
        ):
            routed = await self._route(frame)
            await self.push_frame(routed, direction)
        else:
            await self.push_frame(frame, direction)

    # ── routing logic ─────────────────────────────────────────────────────

    async def _route(self, frame: TranscriptionFrame) -> RoutedTranscriptionFrame:
        """
        Embed the transcript and run k-NN similarity search against the
        seeded intent index.  Returns a RoutedTranscriptionFrame.
        """
        text = frame.text.strip()

        # 1. Embed — run synchronous encoder in thread pool
        loop = asyncio.get_event_loop()
        embedding: np.ndarray = await loop.run_in_executor(
            self._executor,
            self._encoder,
            text,
        )

        # 2. Build the VectorQuery
        from redisvl.query import VectorQuery  # late import — not needed at module level

        query = VectorQuery(
            vector=embedding.astype(np.float32).tolist(),
            vector_field_name="embedding",
            return_fields=["label", "utterance", "vector_distance"],
            num_results=_K,
        )

        # 3. Async similarity search (non-blocking)
        try:
            results = await self._index.query(query)
        except Exception as exc:
            logger.error(f"[ROUTE] Redis query failed: {exc} — defaulting to GENERAL")
            results = []

        # 4. Determine write_signal from nearest neighbour
        write_signal, matched_label, matched_intent, score = self._classify(results)

        # 5. Audit log — the line the tests assert on
        logger.info(
            f'[ROUTE] write_signal={write_signal}'
            f' | label={matched_label}'
            f' | score={score:.3f}'
            f' | matched="{matched_intent}"'
            f' | transcript="{text}"'
        )

        # 6. Build the enriched frame
        return RoutedTranscriptionFrame(
            text=frame.text,
            user_id=frame.user_id,
            timestamp=frame.timestamp,
            write_signal=write_signal,
            similarity_score=score,
            matched_label=matched_label,
            matched_intent=matched_intent,
        )

    @staticmethod
    def _classify(
        results: list[dict],
    ) -> tuple[bool, str, str, float]:
        """
        Parse RedisVL query results into (write_signal, label, intent, score).

        RedisVL returns cosine *distance* (0 = identical, 2 = opposite).
        We convert to similarity = 1 - distance for human readability.
        """
        if not results:
            return False, "UNKNOWN", "", 0.0

        top = results[0]
        label   = top.get("label", "UNKNOWN")
        intent  = top.get("utterance", "")
        distance = float(top.get("vector_distance", 1.0))
        score   = max(0.0, 1.0 - distance)   # cosine similarity ∈ [0, 1]

        write_signal = label == "GAME_ACTION"
        return write_signal, label, intent, score
