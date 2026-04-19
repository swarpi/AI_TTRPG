"""
intents.py
==========
50 labelled D&D intent utterances used to seed the RedisVL vector index.

Each intent is a dict with:
    utterance (str)  — the natural-language phrase
    label     (str)  — "GAME_ACTION" or "GENERAL"

GAME_ACTION → write_signal: True
    The player intends to perform a game-mechanical action that will mutate
    game state (attack, cast spell, move, skill check, use ability, etc.).

GENERAL → write_signal: False
    The utterance is off-topic, a clarifying question, or ambient conversation
    that should NOT trigger a write to the game state.
"""

from __future__ import annotations

INTENTS: list[dict[str, str]] = [
    # ── GAME_ACTION (25) ───────────────────────────────────────────────────
    {"utterance": "I attack the goblin with my longsword",           "label": "GAME_ACTION"},
    {"utterance": "I swing my sword at the orc",                     "label": "GAME_ACTION"},
    {"utterance": "I cast fireball at the group of enemies",         "label": "GAME_ACTION"},
    {"utterance": "I shoot an arrow at the bandit",                  "label": "GAME_ACTION"},
    {"utterance": "I try to pick the lock with my thieves tools",    "label": "GAME_ACTION"},
    {"utterance": "I dash towards the door using my action",        "label": "GAME_ACTION"},
    {"utterance": "I cast healing word on the wounded fighter",      "label": "GAME_ACTION"},
    {"utterance": "I grapple the troll and pin it down",             "label": "GAME_ACTION"},
    {"utterance": "I hide behind the barrel using my bonus action",  "label": "GAME_ACTION"},
    {"utterance": "I throw my handaxe at the skeleton archer",       "label": "GAME_ACTION"},
    {"utterance": "I cast magic missile targeting three enemies",    "label": "GAME_ACTION"},
    {"utterance": "I take the dodge action to avoid attacks",        "label": "GAME_ACTION"},
    {"utterance": "I disarm the pressure plate trap",                "label": "GAME_ACTION"},
    {"utterance": "I persuade the guard to let us through",          "label": "GAME_ACTION"},
    {"utterance": "I search the chest for hidden compartments",      "label": "GAME_ACTION"},
    {"utterance": "I cast shield as a reaction to the attack",       "label": "GAME_ACTION"},
    {"utterance": "I move ten feet to flank the enemy",              "label": "GAME_ACTION"},
    {"utterance": "I use bardic inspiration on the rogue",           "label": "GAME_ACTION"},
    {"utterance": "I divine smite the undead with radiant damage",   "label": "GAME_ACTION"},
    {"utterance": "I wild shape into a brown bear",                  "label": "GAME_ACTION"},
    {"utterance": "I shove the orc off the bridge",                  "label": "GAME_ACTION"},
    {"utterance": "I rage and charge at the orc chieftain",          "label": "GAME_ACTION"},
    {"utterance": "I sneak attack the guard who is distracted",      "label": "GAME_ACTION"},
    {"utterance": "I counterspell the wizard's lightning bolt",      "label": "GAME_ACTION"},
    {"utterance": "I disengage and retreat behind the pillar",       "label": "GAME_ACTION"},

    # ── GENERAL (25) ───────────────────────────────────────────────────────
    {"utterance": "What color is the sky?",                          "label": "GENERAL"},
    {"utterance": "What time is it right now?",                      "label": "GENERAL"},
    {"utterance": "Tell me about the history of ancient Rome",       "label": "GENERAL"},
    {"utterance": "What is two plus two?",                           "label": "GENERAL"},
    {"utterance": "Who is the president of the United States?",      "label": "GENERAL"},
    {"utterance": "What is the best pizza topping?",                 "label": "GENERAL"},
    {"utterance": "How do airplanes stay in the air?",               "label": "GENERAL"},
    {"utterance": "What is the weather forecast for tomorrow?",      "label": "GENERAL"},
    {"utterance": "Can you recommend a good movie to watch?",        "label": "GENERAL"},
    {"utterance": "What is the meaning of life?",                    "label": "GENERAL"},
    {"utterance": "How do you make pasta carbonara?",                "label": "GENERAL"},
    {"utterance": "What language is spoken in Brazil?",              "label": "GENERAL"},
    {"utterance": "Who wrote the play Romeo and Juliet?",            "label": "GENERAL"},
    {"utterance": "What is the capital city of France?",             "label": "GENERAL"},
    {"utterance": "How many planets are in our solar system?",       "label": "GENERAL"},
    {"utterance": "What sport is played with a puck on ice?",        "label": "GENERAL"},
    {"utterance": "Who invented the telephone?",                     "label": "GENERAL"},
    {"utterance": "What year did the second world war end?",         "label": "GENERAL"},
    {"utterance": "How tall is Mount Everest in meters?",            "label": "GENERAL"},
    {"utterance": "What is the speed of light?",                     "label": "GENERAL"},
    {"utterance": "Tell me a funny joke",                            "label": "GENERAL"},
    {"utterance": "What is a good book I should read?",              "label": "GENERAL"},
    {"utterance": "How do computers work?",                          "label": "GENERAL"},
    {"utterance": "What is blockchain technology?",                  "label": "GENERAL"},
    {"utterance": "Can you help me write a professional email?",     "label": "GENERAL"},
]

assert len(INTENTS) == 50, f"Expected 50 intents, got {len(INTENTS)}"
assert sum(1 for i in INTENTS if i["label"] == "GAME_ACTION") == 25
assert sum(1 for i in INTENTS if i["label"] == "GENERAL") == 25
