"""
voice_pipeline/bot.py
======================
Ticket 2.1 — Pipecat WebRTC Echo Bot

Pipeline
--------
    Daily WebRTC in
      → Deepgram Nova-3 STT      (TranscriptionFrame)
      → STTTimestampRecorder     (stamps t_stt)
      → EchoProcessor            (TranscriptionFrame → TTSSpeakFrame)
      → ElevenLabs TTS           (TTSSpeakFrame → TTSAudioRawFrame)
      → TTSStartRecorder         (logs [LATENCY] stt→tts_start N ms)
      → Daily WebRTC out

Startup sequence
----------------
1. If DAILY_ROOM_URL env var is set, use it directly.
2. Otherwise, create an ephemeral Daily room via REST (expires in 1 h).
3. Print the room URL — open it in any browser to join as participant.
4. Bot joins the same room as "GM-Bot" and waits for speech.

Usage
-----
    cp .env.example .env   # fill in API keys
    python bot.py
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from latency_observer import LatencyState, STTTimestampRecorder, TTSStartRecorder
from mcp_bridge import MCPBridge

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Daily room helper
# ---------------------------------------------------------------------------

async def _create_daily_room() -> str:
    """Create an ephemeral Daily room and return its URL."""
    api_key = os.environ["DAILY_API_KEY"]
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "properties": {
                    "exp": int(time.time()) + 3600,  # 1-hour expiry
                    "enable_new_call_ui": True,
                    "start_video_off": True,
                    "start_audio_off": False,
                }
            },
        )
        resp.raise_for_status()
        return resp.json()["url"]


# ---------------------------------------------------------------------------
# Bot entry point
# ---------------------------------------------------------------------------

async def run_bot() -> None:
    # ── 1. Resolve room URL ────────────────────────────────────────────────
    room_url: str = os.getenv("DAILY_ROOM_URL") or await _create_daily_room()
    logger.info("━" * 60)
    logger.info(f"🎲  Room URL : {room_url}")
    logger.info("   Open this URL in your browser and click 'Join'")
    logger.info("   Speak something → hear it echoed back via TTS")
    logger.info("━" * 60)

    # ── 2. Transport ───────────────────────────────────────────────────────
    transport = DailyTransport(
        room_url,
        None,           # token — None for URL-only access rooms
        "GM-Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,      # Daily in-browser VAD for cleaner audio
            vad_stop_secs=0.4,    # 400 ms silence → end-of-utterance
        ),
    )

    # ── 3. STT ─────────────────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        # Nova-3 is Deepgram's most accurate streaming model (2025)
        model="nova-3",
        language="en-US",
        smart_format=True,
        interim_results=False,  # Only final transcripts → EchoProcessor
        endpointing=300,        # ms of silence to finalize an utterance
    )

    # ── 4. TTS ─────────────────────────────────────────────────────────────
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        # output_format="pcm_24000" — Daily expects 16 kHz PCM; ElevenLabs default is fine
    )

    # ── 5. Latency tracking ────────────────────────────────────────────────
    state = LatencyState()
    stt_ts = STTTimestampRecorder(state)
    tts_ts = TTSStartRecorder(state)

    # ── 6. Semantic Router (Ticket 2.2) ──────────────────────────────────────
    from seed_redis import REDIS_URL, _build_schema, build_encoder
    from redisvl.index import AsyncSearchIndex
    from redisvl.schema import IndexSchema
    from semantic_router import SemanticRouter
    from ui_lock_processor import UILockProcessor

    schema = IndexSchema.from_dict(_build_schema())
    index = AsyncSearchIndex(schema, url=REDIS_URL)
    router = SemanticRouter(index=index, encoder=build_encoder())

    # ── 7. MCP Bridge (Ticket 3.2) ───────────────────────────────────────────
    mcp_bridge = MCPBridge()
    await mcp_bridge.connect()

    # ── 8. LLM setup ───────────────────────────────────────────────────────
    from llm_factory import create_llm_service
    llm, context = create_llm_service()

    llm.register_function(
        "dnd_resolve_melee_attack",
        mcp_bridge.dnd_resolve_melee_attack,
        description="Resolve a D&D melee attack mathematically.",
        parameters={
            "type": "object",
            "properties": {
                "attacker_id": {"type": "string", "description": "e.g. player_1, orc_captain"},
                "target_id": {"type": "string", "description": "e.g. goblin_1, player_2"},
                "weapon": {"type": "string", "description": "e.g. longsword, dagger"}
            },
            "required": ["attacker_id", "target_id", "weapon"]
        }
    )

    context_aggregator = llm.create_context_aggregator(context)

    # ── 9. Pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),   # Daily WebRTC audio → AudioRawFrame
            stt,                 # AudioRawFrame → TranscriptionFrame
            stt_ts,              # stamp t_stt (passthrough)
            router,              # TranscriptionFrame → RoutedTranscriptionFrame
            UILockProcessor(),   # RoutedTranscriptionFrame → (emits lock) → RoutedTranscriptionFrame
            context_aggregator.user(),
            llm,                 # Generates TTSSpeakFrames
            tts,                 # TTSSpeakFrame → TTSAudioRawFrame
            tts_ts,              # log latency on first chunk (passthrough)
            transport.output(),  # TTSAudioRawFrame → Daily WebRTC audio
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── 7. Event handlers ──────────────────────────────────────────────────

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):  # type: ignore[misc]
        logger.info(f"🎭  Participant joined — start speaking!")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):  # type: ignore[misc]
        logger.info("👋  Participant left — stopping bot")
        await task.cancel()

    # ── 10. Run ─────────────────────────────────────────────────────────────
    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)
    
    # Cleanup MCP SSE connection
    await mcp_bridge.disconnect()

if __name__ == "__main__":
    asyncio.run(run_bot())
