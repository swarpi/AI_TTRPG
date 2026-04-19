"""
llm_factory.py
==============
Ticket 3.3 — LLM Factory Pattern

Provides a dynamic LLM service constructor driven by the `LLM_PROVIDER`
environment variable (supports 'anthropic' and 'openai').
"""

import os
from loguru import logger

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

_GM_SYSTEM_PROMPT = (
    "You are a D&D Game Master. "
    "When a player declares an attack, call the dnd_resolve_melee_attack tool. "
    "Read the JSON result carefully (roll, hit_bonus, ac, is_hit, damage, target_hp) "
    "and descriptively narrate the exact mathematical outcome in 2-3 short, punchy sentences. "
    "If the target_hp drops to 0, narrate a fatal blow."
)

def create_llm_service():
    """
    Reads LLM_PROVIDER from env (default: 'anthropic') and returns
    (LLMService, OpenAILLMContext).
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    
    if provider == "openai":
        logger.info("🤖 Using OpenAI LLM Service (gpt-4o)")
        from pipecat.services.openai import OpenAILLMService
        
        # Verify key exists early
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in environment.")
            
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o",
        )
    else:
        logger.info("🤖 Using Anthropic LLM Service (claude-3-5-sonnet-20241022)")
        from pipecat.services.anthropic import AnthropicLLMService
        
        # Verify key exists early
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is not set in environment.")

        llm = AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-3-5-sonnet-20241022",
        )

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": _GM_SYSTEM_PROMPT}]
    )

    return llm, context
