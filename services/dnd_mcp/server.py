#!/usr/bin/env python3
"""
dnd_mcp/server.py
=================
AI-TTRPG FastMCP service — D&D 5e rules engine wrapper.

Ticket 1.1:
  - Stand up a FastMCP 3 server (streamable-HTTP transport, port 8000).
  - Wrap the dnd-5e-core library so all tools can call it.
  - Configure OpenTelemetry to emit basic lifecycle spans to the console.

OpenTelemetry must be initialised *before* FastMCP is imported so that
its no-op API hooks are replaced by our ConsoleSpanExporter.
"""

# ---------------------------------------------------------------------------
# 1. OpenTelemetry bootstrap  (MUST come before any FastMCP import)
# ---------------------------------------------------------------------------
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

_resource = Resource.create({"service.name": "dnd-mcp", "service.version": "0.1.0"})
_provider = TracerProvider(resource=_resource)

# Console exporter — prints human-readable span JSON to stdout.
# Use SimpleSpanProcessor so spans flush synchronously (great for local dev).
_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(_provider)

# ---------------------------------------------------------------------------
# 2. Standard library & third-party imports
# ---------------------------------------------------------------------------
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from opentelemetry.trace import get_tracer
from pydantic import BaseModel, ConfigDict, Field, field_validator

# FastMCP imports (after OTEL bootstrap)
from fastmcp import FastMCP, Context

# ---------------------------------------------------------------------------
# 3. dnd-5e-core imports
#    The library is UI-agnostic: it exposes Python classes / functions that
#    we call inside MCP tools.  We import eagerly so missing-package errors
#    surface at startup rather than at first tool invocation.
# ---------------------------------------------------------------------------
from dnd_5e_core.data.loaders import simple_character_generator
from dnd_5e_core import load_monster
from dnd_5e_core.combat import CombatSystem
from dnd_5e_core.equipment import get_magic_item

# Ticket 1.2 — pure combat logic (no OTEL/FastMCP deps, fully unit-testable)
from combat import resolve_melee_attack as _resolve_melee_attack, WEAPON_TABLE, GAME_STATE

# ---------------------------------------------------------------------------
# 4. Module-level constants
# ---------------------------------------------------------------------------
SERVER_NAME = "dnd_mcp"
SERVER_PORT = 8000

_logger = logging.getLogger(SERVER_NAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

_tracer = get_tracer(SERVER_NAME)

# ---------------------------------------------------------------------------
# 5. Server lifespan — initialise / teardown shared state
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(server: FastMCP):
    """
    Set up shared resources that tools can access via ctx.request_context.
    Currently pre-warms the CombatSystem so tools don't pay init cost.
    """
    with _tracer.start_as_current_span("dnd_mcp.startup"):
        _logger.info("🐉  dnd_mcp starting — loading D&D 5e rules engine …")
        combat_system = CombatSystem(verbose=False)
        _logger.info("✅  dnd_mcp ready on port %d", SERVER_PORT)

    yield {"combat_system": combat_system}

    with _tracer.start_as_current_span("dnd_mcp.shutdown"):
        _logger.info("👋  dnd_mcp shutting down")


# ---------------------------------------------------------------------------
# 6. FastMCP server initialisation
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name=SERVER_NAME,
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# 7. Pydantic input models
# ---------------------------------------------------------------------------

class CharacterCreateInput(BaseModel):
    """Input for creating a D&D 5e character."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    level: int = Field(
        ...,
        description="Character level (1–20)",
        ge=1,
        le=20,
    )
    race: str = Field(
        ...,
        description=(
            "Character race (e.g., 'human', 'elf', 'dwarf', 'halfling', "
            "'dragonborn', 'gnome', 'half-elf', 'half-orc', 'tiefling')"
        ),
        min_length=2,
        max_length=40,
    )
    char_class: str = Field(
        ...,
        description=(
            "Character class (e.g., 'fighter', 'wizard', 'rogue', 'cleric', "
            "'ranger', 'paladin', 'bard', 'druid', 'barbarian', 'monk', "
            "'warlock', 'sorcerer')"
        ),
        min_length=2,
        max_length=40,
    )
    name: str = Field(
        ...,
        description="Character name (e.g., 'Conan', 'Gandalf', 'Legolas')",
        min_length=1,
        max_length=60,
    )

    @field_validator("race", "char_class")
    @classmethod
    def to_lower(cls, v: str) -> str:
        return v.lower().strip()


class MonsterLoadInput(BaseModel):
    """Input for loading a D&D 5e monster by name."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    monster_name: str = Field(
        ...,
        description=(
            "Monster name as it appears in the SRD (e.g., 'orc', 'goblin', "
            "'dragon', 'skeleton', 'zombie', 'beholder'). Case-insensitive."
        ),
        min_length=2,
        max_length=80,
    )

    @field_validator("monster_name")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.lower().strip()


class MagicItemInput(BaseModel):
    """Input for retrieving a magic item by its slug."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    item_slug: str = Field(
        ...,
        description=(
            "Magic item slug (e.g., 'ring-of-protection', 'wand-of-magic-missiles', "
            "'cloak-of-invisibility', 'bag-of-holding'). Use hyphens, lowercase."
        ),
        min_length=2,
        max_length=80,
    )

    @field_validator("item_slug")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.lower().strip().replace(" ", "-")


# ---------------------------------------------------------------------------
# 8. Utility helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any, indent: int = 2) -> str:
    """Serialize an object to JSON, falling back to repr() for non-serialisable types."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except Exception:
        return repr(obj)


def _char_summary(char: Any) -> Dict[str, Any]:
    """Extract a clean dict summary from a dnd_5e_core Character object."""
    return {
        "name": getattr(char, "name", "unknown"),
        "race": getattr(char, "race", "unknown"),
        "char_class": getattr(char, "char_class", "unknown"),
        "level": getattr(char, "level", 0),
        "hp": getattr(char, "hp", 0),
        "max_hp": getattr(char, "max_hp", getattr(char, "hp", 0)),
        "ac": getattr(char, "ac", 10),
        "abilities": {
            "str": getattr(char, "strength", None),
            "dex": getattr(char, "dexterity", None),
            "con": getattr(char, "constitution", None),
            "int": getattr(char, "intelligence", None),
            "wis": getattr(char, "wisdom", None),
            "cha": getattr(char, "charisma", None),
        },
        "proficiency_bonus": getattr(char, "proficiency_bonus", None),
        "class_abilities": [
            getattr(a, "name", str(a))
            for a in getattr(char, "class_abilities", [])
        ],
        "racial_traits": [
            getattr(t, "name", str(t))
            for t in getattr(char, "racial_traits", [])
        ],
    }


def _monster_summary(monster: Any) -> Dict[str, Any]:
    """Extract a clean dict summary from a dnd_5e_core Monster object."""
    return {
        "name": getattr(monster, "name", "unknown"),
        "cr": getattr(monster, "challenge_rating", getattr(monster, "cr", "?")),
        "hp": getattr(monster, "hp", 0),
        "ac": getattr(monster, "ac", 10),
        "type": getattr(monster, "creature_type", getattr(monster, "type", "?")),
        "size": getattr(monster, "size", "?"),
        "attacks": [
            getattr(a, "name", str(a))
            for a in getattr(monster, "attacks", [])
        ],
        "special_abilities": [
            getattr(a, "name", str(a))
            for a in getattr(monster, "special_abilities", [])
        ],
    }


# ---------------------------------------------------------------------------
# 9. Tool definitions
# ---------------------------------------------------------------------------

@mcp.tool(
    name="dnd_create_character",
    annotations={
        "title": "Create D&D 5e Character",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def dnd_create_character(
    level: int,
    race: str,
    char_class: str,
    name: str,
    ctx: Context,
) -> str:
    """Generate a fully-featured D&D 5e character using the dnd-5e-core rules engine.

    Creates a character with the appropriate class abilities, racial traits,
    hit points, armour class, ability scores, and proficiency bonus for the
    specified level. All calculations follow the 5e SRD rules.

    Args:
        level: Character level 1-20 (e.g. 5)
        race: Race slug — 'human', 'elf', 'dwarf', 'halfling', 'dragonborn',
              'gnome', 'half-elf', 'half-orc', 'tiefling'
        char_class: Class slug — 'fighter', 'wizard', 'rogue', 'cleric',
                    'ranger', 'paladin', 'bard', 'druid', 'barbarian',
                    'monk', 'warlock', 'sorcerer'
        name: Character name (e.g. 'Gimli', 'Gandalf')
        ctx: FastMCP context (injected automatically)

    Returns:
        str: JSON-formatted character sheet including stats, abilities and traits.

    Examples:
        - "Create a level 5 dwarf fighter named Gimli"
        - "Make a level 3 elf wizard named Elrond"
    """
    race = race.lower().strip()
    char_class = char_class.lower().strip()
    with _tracer.start_as_current_span(
        "dnd_mcp.tool.create_character",
        attributes={
            "character.name": name,
            "character.level": level,
            "character.race": race,
            "character.class": char_class,
        },
    ):
        await ctx.info(
            f"Creating character: {name} (level {level} {race} {char_class})"
        )
        try:
            char = simple_character_generator(level, race, char_class, name)
            summary = _char_summary(char)
            return _safe_json({"status": "created", "character": summary})
        except Exception as exc:
            _logger.exception("Error creating character: %s", exc)
            return f"Error: Could not create character — {exc}. Check that race and class are valid 5e SRD values."


@mcp.tool(
    name="dnd_load_monster",
    annotations={
        "title": "Load D&D 5e Monster",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dnd_load_monster(monster_name: str, ctx: Context) -> str:
    """Load a D&D 5e monster's complete stat block from the bundled offline database.

    Retrieves 332+ monsters including all SRD creatures. Returns a structured
    JSON stat block with CR, HP, AC, attacks, and special abilities.

    Args:
        monster_name: SRD monster name — lowercase, e.g. 'goblin', 'orc',
                      'skeleton', 'zombie', 'beholder', 'dragon'
        ctx: FastMCP context (injected automatically)

    Returns:
        str: JSON-formatted monster stat block.

    Examples:
        - "Get stats for a goblin"
        - "Load the beholder monster"
        - "What are the stats for an orc?"
    """
    monster_name = monster_name.lower().strip()
    with _tracer.start_as_current_span(
        "dnd_mcp.tool.load_monster",
        attributes={"monster.name": monster_name},
    ):
        await ctx.info(f"Loading monster: {monster_name}")
        try:
            monster = load_monster(monster_name)
            summary = _monster_summary(monster)
            return _safe_json({"status": "found", "monster": summary})
        except Exception as exc:
            _logger.exception("Error loading monster '%s': %s", monster_name, exc)
            return (
                f"Error: Monster '{monster_name}' not found. "
                "Try lowercase SRD names like 'goblin', 'orc', 'skeleton', 'zombie'. "
                f"Details: {exc}"
            )


@mcp.tool(
    name="dnd_get_magic_item",
    annotations={
        "title": "Get D&D 5e Magic Item",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dnd_get_magic_item(item_slug: str, ctx: Context) -> str:
    """Retrieve a D&D 5e magic item's properties from the bundled offline database (49 items).

    Returns item rarity, attunement requirements, charges, and magical properties.

    Args:
        item_slug: Hyphenated item slug, e.g. 'ring-of-protection',
                   'wand-of-magic-missiles', 'bag-of-holding',
                   'cloak-of-invisibility'. Spaces are auto-converted to hyphens.
        ctx: FastMCP context (injected automatically)

    Returns:
        str: JSON-formatted magic item properties.

    Examples:
        - "Get stats for ring of protection"
        - "What does a wand of magic missiles do?"
        - "Describe the bag of holding"
    """
    item_slug = item_slug.lower().strip().replace(" ", "-")
    with _tracer.start_as_current_span(
        "dnd_mcp.tool.get_magic_item",
        attributes={"item.slug": item_slug},
    ):
        await ctx.info(f"Looking up magic item: {item_slug}")
        try:
            item = get_magic_item(item_slug)
            item_data = {
                "name": getattr(item, "name", item_slug),
                "rarity": getattr(item, "rarity", "unknown"),
                "requires_attunement": getattr(item, "requires_attunement", False),
                "charges": getattr(item, "charges", None),
                "properties": getattr(item, "properties", []),
                "description": getattr(item, "description", ""),
            }
            return _safe_json({"status": "found", "item": item_data})
        except Exception as exc:
            _logger.exception("Error loading magic item '%s': %s", item_slug, exc)
            return (
                f"Error: Magic item '{item_slug}' not found. "
                "Use hyphenated slugs like 'ring-of-protection', 'wand-of-magic-missiles'. "
                f"Details: {exc}"
            )


# ---------------------------------------------------------------------------
# 10. Ticket 1.2 — resolve_melee_attack tool
# ---------------------------------------------------------------------------
_VALID_WEAPONS = ", ".join(sorted(WEAPON_TABLE.keys()))
_VALID_IDS = ", ".join(
    sorted(list(GAME_STATE["players"]) + list(GAME_STATE["npcs"]))
)


@mcp.tool(
    name="dnd_resolve_melee_attack",
    annotations={
        "title": "Resolve Melee Attack",
        "readOnlyHint": False,      # mutates target HP
        "destructiveHint": True,   # can reduce target to 0 HP
        "idempotentHint": False,   # repeated calls produce different rolls
        "openWorldHint": False,    # operates only on in-session game state
    },
)
async def dnd_resolve_melee_attack(
    attacker_id: str,
    target_id: str,
    weapon: str,
    ctx: Context,
) -> str:
    """Resolve a single D&D 5e melee attack between two entities in the current encounter.

    Executes the full 5e attack pipeline:
      1. Roll a virtual d20.
      2. Add the attacker's STR modifier + proficiency bonus.
      3. Compare the total against the target's AC.
      4. If hit: roll weapon damage + STR modifier and deduct from target HP.

    The returned JSON observation is suitable for direct consumption by an LLM
    as a tool result, letting it narrate the outcome.

    Args:
        attacker_id: ID of the attacking entity.
            Players: player_1, player_2, player_3
            NPCs:    goblin_1, orc_1, orc_captain, ancient_dragon
        target_id: ID of the defending entity (must differ from attacker_id).
            Same valid IDs as attacker_id.
        weapon: Melee weapon slug. Valid values:
            battleaxe, club, dagger, greataxe, greatsword, handaxe,
            longsword, maul, rapier, scimitar, shortsword, unarmed, warhammer
        ctx: FastMCP context (injected automatically).

    Returns:
        str: JSON object with schema:
            {
                "roll":      int,   # raw d20 result (1-20)
                "hit_bonus": int,   # STR modifier + proficiency bonus
                "total":     int,   # roll + hit_bonus
                "ac":        int,   # target's armour class
                "is_hit":    bool,  # true if total >= ac
                "damage":    int,   # damage dealt (0 on miss)
                "target_hp": int,   # target HP after attack
                "attacker":  str,   # display name of attacker
                "target":    str    # display name of target
            }

    Examples:
        - player_1 attacks goblin_1 with longsword
        - orc_captain attacks player_2 with greataxe
    """
    attacker_id = attacker_id.strip()
    target_id   = target_id.strip()
    weapon      = weapon.strip()

    # ── Strict schema validation before touching the engine ────────────────
    errors: list[str] = []
    if attacker_id not in (list(GAME_STATE["players"]) + list(GAME_STATE["npcs"])):
        errors.append(f"Unknown attacker_id '{attacker_id}'. Valid IDs: {_VALID_IDS}")
    if target_id not in (list(GAME_STATE["players"]) + list(GAME_STATE["npcs"])):
        errors.append(f"Unknown target_id '{target_id}'. Valid IDs: {_VALID_IDS}")
    if attacker_id == target_id:
        errors.append("attacker_id and target_id must be different entities.")
    if weapon.lower().strip().replace(" ", "").replace("_", "") not in WEAPON_TABLE:
        errors.append(f"Unknown weapon '{weapon}'. Valid weapons: {_VALID_WEAPONS}")
    if errors:
        return _safe_json({"error": "Input validation failed", "details": errors})

    with _tracer.start_as_current_span(
        "dnd_mcp.tool.resolve_melee_attack",
        attributes={
            "attack.attacker_id": attacker_id,
            "attack.target_id":   target_id,
            "attack.weapon":      weapon,
        },
    ):
        await ctx.info(
            f"Resolving melee attack: {attacker_id} → {target_id} with {weapon}"
        )
        try:
            result = _resolve_melee_attack(attacker_id, target_id, weapon)
            _logger.info(
                "Attack resolved: %s → %s | roll=%d bonus=%d total=%d ac=%d "
                "is_hit=%s damage=%d hp_remaining=%d",
                result["attacker"], result["target"],
                result["roll"], result["hit_bonus"], result["total"],
                result["ac"], result["is_hit"], result["damage"], result["target_hp"],
            )
            return _safe_json(result)
        except (KeyError, ValueError) as exc:
            return _safe_json({"error": str(exc)})
        except Exception as exc:
            _logger.exception("Unexpected error in resolve_melee_attack: %s", exc)
            return _safe_json({"error": f"Internal error: {exc}"})


# ---------------------------------------------------------------------------
# 11. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Streamable HTTP transport — accessible at http://localhost:8000/mcp/
    mcp.run(transport="streamable-http", host="0.0.0.0", port=SERVER_PORT)
