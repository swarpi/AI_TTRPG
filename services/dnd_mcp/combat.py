"""
combat.py
=========
Pure-Python melee attack resolution for the AI-TTRPG POC.

No FastMCP or OTEL dependencies — this module can be imported and unit-tested
in isolation without starting the server.

Architecture note
-----------------
GAME_STATE is the in-process mock of what will eventually be a persistent store
(Cloudflare Durable Object / Postgres).  For the POC it is a module-level dict
that is mutated on each successful hit — i.e. HP changes are reflected within
a single server session.

Sections
--------
1. WEAPON_TABLE  — damage dice for supported melee weapons
2. GAME_STATE    — mock Players (Users.md equivalent) and NPCs (NPCs.md)
3. Helpers       — modifier calc, dice wrappers (patchable in tests)
4. resolve_melee_attack() — the main entry point called by the MCP tool
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# 1. Weapon table
#    Each entry: (damage_die_sides, number_of_dice, damage_type)
# ---------------------------------------------------------------------------
WEAPON_TABLE: Dict[str, Dict[str, Any]] = {
    "dagger":       {"die": 4,  "count": 1, "type": "piercing"},
    "handaxe":      {"die": 6,  "count": 1, "type": "slashing"},
    "shortsword":   {"die": 6,  "count": 1, "type": "piercing"},
    "scimitar":     {"die": 6,  "count": 1, "type": "slashing"},
    "longsword":    {"die": 8,  "count": 1, "type": "slashing"},
    "rapier":       {"die": 8,  "count": 1, "type": "piercing"},
    "battleaxe":    {"die": 8,  "count": 1, "type": "slashing"},
    "warhammer":    {"die": 8,  "count": 1, "type": "bludgeoning"},
    "greataxe":     {"die": 12, "count": 1, "type": "slashing"},
    "greatsword":   {"die": 6,  "count": 2, "type": "slashing"},
    "maul":         {"die": 6,  "count": 2, "type": "bludgeoning"},
    "club":         {"die": 4,  "count": 1, "type": "bludgeoning"},
    "unarmed":      {"die": 1,  "count": 1, "type": "bludgeoning"},  # 1 + STR mod
}

# ---------------------------------------------------------------------------
# 2. Mock game state  (replaces file-system reads of Users.md / NPCs.md)
#
#    Each entity has the minimum fields needed for melee resolution:
#      str_score        — Strength ability score (modifier = (score-10)//2)
#      proficiency_bonus — Added to attack roll
#      ac               — Armour Class of the target
#      hp / max_hp      — Current and maximum hit points
#
#    HP is mutable; all other fields are treated as read-only during a session.
# ---------------------------------------------------------------------------
GAME_STATE: Dict[str, Any] = {
    # ── Players (Users.md) ─────────────────────────────────────────────────
    "players": {
        "player_1": {
            "name": "Conan",
            "race": "human",
            "class": "fighter",
            "level": 5,
            "str_score": 18,   # +4 modifier
            "proficiency_bonus": 3,
            "ac": 16,          # chain mail + shield
            "hp": 52,
            "max_hp": 52,
        },
        "player_2": {
            "name": "Elara",
            "race": "elf",
            "class": "rogue",
            "level": 5,
            "str_score": 12,   # +1 modifier
            "proficiency_bonus": 3,
            "ac": 14,          # leather armour + DEX
            "hp": 33,
            "max_hp": 33,
        },
        "player_3": {
            "name": "Torven",
            "race": "dwarf",
            "class": "cleric",
            "level": 5,
            "str_score": 14,   # +2 modifier
            "proficiency_bonus": 3,
            "ac": 18,          # plate armour
            "hp": 44,
            "max_hp": 44,
        },
    },
    # ── NPCs (NPCs.md) ─────────────────────────────────────────────────────
    "npcs": {
        # Low-AC target — easy to hit
        "goblin_1": {
            "name": "Skrix the Goblin",
            "cr": 0.25,
            "str_score": 8,    # -1 modifier
            "proficiency_bonus": 2,
            "ac": 12,
            "hp": 7,
            "max_hp": 7,
        },
        # Medium-AC target — moderately hard to hit
        "orc_1": {
            "name": "Grak Ironjaw",
            "cr": 0.5,
            "str_score": 16,   # +3 modifier
            "proficiency_bonus": 2,
            "ac": 13,
            "hp": 15,
            "max_hp": 15,
        },
        "orc_captain": {
            "name": "Drek the Captain",
            "cr": 2,
            "str_score": 18,   # +4 modifier
            "proficiency_bonus": 2,
            "ac": 16,
            "hp": 42,
            "max_hp": 42,
        },
        # High-AC target — very hard to hit
        "ancient_dragon": {
            "name": "Ignathrix the Ancient",
            "cr": 24,
            "str_score": 30,   # +10 modifier
            "proficiency_bonus": 7,
            "ac": 22,
            "hp": 546,
            "max_hp": 546,
        },
    },
}

# ---------------------------------------------------------------------------
# 3. Helpers
#    Dice functions are broken out so tests can mock them individually.
# ---------------------------------------------------------------------------

def _str_modifier(str_score: int) -> int:
    """Return the STR ability modifier for a given score (standard 5e formula)."""
    return (str_score - 10) // 2


def _roll_d20() -> int:
    """Roll a single d20. Extracted so unit tests can patch it."""
    return random.randint(1, 20)


def _roll_damage(weapon: Dict[str, Any], str_mod: int) -> int:
    """Roll weapon damage and add the STR modifier (minimum 1)."""
    raw = sum(random.randint(1, weapon["die"]) for _ in range(weapon["count"]))
    return max(1, raw + str_mod)


def _lookup_entity(entity_id: str) -> Tuple[Dict[str, Any], str]:
    """
    Find an entity by ID in GAME_STATE.

    Returns (entity_dict, pool_name) where pool_name is 'players' or 'npcs'.
    Raises KeyError with a helpful message if not found.
    """
    for pool in ("players", "npcs"):
        if entity_id in GAME_STATE[pool]:
            return GAME_STATE[pool][entity_id], pool

    valid_ids = sorted(
        list(GAME_STATE["players"]) + list(GAME_STATE["npcs"])
    )
    raise KeyError(
        f"Entity '{entity_id}' not found. "
        f"Valid IDs: {valid_ids}"
    )


# ---------------------------------------------------------------------------
# 4. Main resolution function
#    Called by the MCP tool wrapper in server.py.
#    Does NOT mutate GAME_STATE directly — returns a copy-safe result dict,
#    but DOES update hp in-place to persist HP changes across calls.
# ---------------------------------------------------------------------------

def resolve_melee_attack(
    attacker_id: str,
    target_id: str,
    weapon: str,
    *,
    _state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Resolve a single melee attack according to D&D 5e rules.

    Pipeline
    --------
    1. Look up attacker & target in the game state.
    2. Validate the weapon slug against WEAPON_TABLE.
    3. Roll d20 + STR modifier + proficiency bonus.
    4. Compare attack total against target AC → is_hit.
    5. If hit: roll weapon damage + STR modifier, deduct from target HP.
    6. Return the observation dict and persist HP change.

    Parameters
    ----------
    attacker_id : str
        ID of the attacking entity (e.g. 'player_1', 'orc_1').
    target_id : str
        ID of the defending entity.
    weapon : str
        Weapon slug from WEAPON_TABLE (e.g. 'longsword', 'dagger').
    _state : dict, optional
        Alternate game state dict (used by tests to provide an isolated copy).

    Returns
    -------
    dict with keys:
        roll       (int)  — raw d20 result (1–20)
        hit_bonus  (int)  — STR modifier + proficiency bonus added to roll
        total      (int)  — roll + hit_bonus (the full attack total)
        ac         (int)  — target's armour class
        is_hit     (bool) — whether total >= ac
        damage     (int)  — damage dealt (0 if miss)
        target_hp  (int)  — target's HP *after* attack resolves
        attacker   (str)  — display name of attacker
        target     (str)  — display name of target

    Raises
    ------
    KeyError  — unknown attacker_id, target_id, or weapon slug
    ValueError — attacker and target are the same entity
    """
    state = _state if _state is not None else GAME_STATE

    # ── Validate inputs ────────────────────────────────────────────────────
    weapon_slug = weapon.lower().strip().replace(" ", "").replace("_", "")
    if weapon_slug not in WEAPON_TABLE:
        valid = sorted(WEAPON_TABLE.keys())
        raise KeyError(
            f"Unknown weapon '{weapon}'. "
            f"Valid weapons: {valid}"
        )

    # Use the shared lookup but resolve against the provided state
    def _lookup(eid: str) -> Dict[str, Any]:
        for pool in ("players", "npcs"):
            if eid in state[pool]:
                return state[pool][eid]
        valid_ids = sorted(list(state["players"]) + list(state["npcs"]))
        raise KeyError(
            f"Entity '{eid}' not found. Valid IDs: {valid_ids}"
        )

    if attacker_id == target_id:
        raise ValueError("Attacker and target must be different entities.")

    attacker = _lookup(attacker_id)
    target   = _lookup(target_id)
    wpn      = WEAPON_TABLE[weapon_slug]

    # ── Roll ───────────────────────────────────────────────────────────────
    str_mod    = _str_modifier(attacker["str_score"])
    prof_bonus = attacker["proficiency_bonus"]
    hit_bonus  = str_mod + prof_bonus

    d20_roll   = _roll_d20()
    total      = d20_roll + hit_bonus
    ac         = target["ac"]
    is_hit     = total >= ac

    # ── Damage ─────────────────────────────────────────────────────────────
    damage = 0
    if is_hit:
        damage = _roll_damage(wpn, str_mod)
        # Persist HP change (minimum 0)
        target["hp"] = max(0, target["hp"] - damage)

    return {
        "roll":      d20_roll,
        "hit_bonus": hit_bonus,
        "total":     total,
        "ac":        ac,
        "is_hit":    is_hit,
        "damage":    damage,
        "target_hp": target["hp"],
        "attacker":  attacker["name"],
        "target":    target["name"],
    }
