"""
seed_redis.py
=============
One-shot script to create and seed the RedisVL vector index with 50 D&D intents.

Run once before starting the bot:

    python seed_redis.py

What it does
------------
1. Loads the 50 labelled utterances from intents.py.
2. Encodes all utterances with fastembed (BAAI/bge-small-en-v1.5, 384 dims).
3. Creates (or recreates) the "dnd_intents" RedisVL index.
4. Upserts all intent vectors in a single batch.

Redis must be reachable at REDIS_URL (default: redis://localhost:6379).
Start the container first:

    docker compose up -d redis

Requirements
------------
    pip install -e ".[dev]"   # redisvl and fastembed are included
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from intents import INTENTS  # 50 labelled utterances

# ---------------------------------------------------------------------------
# Index schema constants
# ---------------------------------------------------------------------------

INDEX_NAME    = "dnd_intents"
EMBEDDING_DIM = 384   # BAAI/bge-small-en-v1.5 output dimension
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379")


def _build_schema() -> dict:
    return {
        "index": {
            "name":    INDEX_NAME,
            "prefix":  "intent",
            "storage_type": "hash",
        },
        "fields": [
            {"type": "tag",  "name": "label"},
            {"type": "text", "name": "utterance"},
            {
                "type": "vector",
                "name": "embedding",
                "attrs": {
                    "dims":            EMBEDDING_DIM,
                    "distance_metric": "cosine",
                    "algorithm":       "flat",
                    "datatype":        "float32",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Encoder — synchronous, called once for the batch
# ---------------------------------------------------------------------------

def build_encoder():
    """Lazy-load fastembed so the import is only paid when seeding."""
    from fastembed import TextEmbedding
    logger.info("Loading fastembed model BAAI/bge-small-en-v1.5 …")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def encoder(text: str) -> np.ndarray:
        return next(model.embed([text])).astype(np.float32)

    return encoder


def encode_batch(utterances: list[str]) -> list[np.ndarray]:
    """Encode all utterances in one pass (faster than one-by-one)."""
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return [emb.astype(np.float32) for emb in model.embed(utterances)]


# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------

async def seed() -> None:
    from redisvl.index import AsyncSearchIndex
    from redisvl.schema import IndexSchema

    logger.info(f"Connecting to Redis at {REDIS_URL} …")

    schema = IndexSchema.from_dict(_build_schema())
    index = AsyncSearchIndex(schema, url=REDIS_URL, overwrite=True)

    # Connect and (re)create the index — overwrite=True drops existing data
    await index.create(overwrite=True)
    logger.info(f"Index '{INDEX_NAME}' created (overwrite=True).")

    # Encode all 50 utterances in a single batch
    utterances = [i["utterance"] for i in INTENTS]
    logger.info(f"Encoding {len(utterances)} utterances with fastembed …")
    embeddings = encode_batch(utterances)

    # Build records for redisvl bulk upsert
    records = [
        {
            "label":     intent["label"],
            "utterance": intent["utterance"],
            # redisvl stores float32 vectors as raw bytes in Hash fields
            "embedding": emb.tobytes(),
        }
        for intent, emb in zip(INTENTS, embeddings)
    ]

    await index.load(records)
    logger.info(f"✅  Seeded {len(records)} intent vectors into '{INDEX_NAME}'.")

    # Quick sanity check — retrieve with a known game-action query
    from redisvl.query import VectorQuery
    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    test_vec = next(model.embed(["I hit the goblin with my axe"])).astype(np.float32)

    q = VectorQuery(
        vector=test_vec.tolist(),
        vector_field_name="embedding",
        return_fields=["label", "utterance", "vector_distance"],
        num_results=1,
    )
    results = await index.query(q)
    if results:
        hit = results[0]
        logger.info(
            f"Sanity check — nearest to 'I hit the goblin':"
            f" label={hit['label']}"
            f" dist={float(hit['vector_distance']):.3f}"
            f" utterance=\"{hit['utterance']}\""
        )
    else:
        logger.warning("Sanity check returned no results.")

    await index.disconnect()


if __name__ == "__main__":
    asyncio.run(seed())
