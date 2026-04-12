"""
tests/test_resolve_melee_attack.py
====================================
Unit tests for Ticket 1.2: resolve_melee_attack tool schema + engine.

Tests run against combat.py directly — no MCP server required.
Dice rolls are mocked so results are fully deterministic.

Test matrix
-----------
1. High-AC target (ancient_dragon, AC 22) + guaranteed miss roll → is_hit: false, damage: 0
2. Low-AC target  (goblin_1, AC 12) + guaranteed hit roll  → is_hit: true, damage >= 1
3. Schema — ALL required keys present and correct types
4. Input validation — unknown attacker, target, weapon, same-entity attack
5. STR modifier math — verify hit_bonus and total computed correctly
6. HP persistence — target HP decreases after a hit
7. Natural 20 — still registers as a hit (total = 20 + bonus >= any normal AC)
8. Weapon variety — shortsword, greataxe, greatsword all produce valid damage
"""

from __future__ import annotations

import copy
import sys
import os

# Ensure the package root is importable when running from any working dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch

from combat import (
    GAME_STATE,
    WEAPON_TABLE,
    resolve_melee_attack,
    _str_modifier,
    _roll_d20,
    _roll_damage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_state() -> dict:
    """Return a deep-copy of GAME_STATE so tests don't bleed into each other."""
    return copy.deepcopy(GAME_STATE)


def attack(
    attacker_id: str,
    target_id: str,
    weapon: str,
    d20_result: int,
    state: dict | None = None,
) -> dict:
    """
    Convenience wrapper: run resolve_melee_attack with a fixed d20 roll.
    Damage dice are left random (seeded implicitly by d20 mock side-effect).
    """
    st = state or fresh_state()
    with patch("combat._roll_d20", return_value=d20_result):
        return resolve_melee_attack(attacker_id, target_id, weapon, _state=st)


# ---------------------------------------------------------------------------
# 1. High-AC target must miss on a low roll
# ---------------------------------------------------------------------------

class TestHighAcTargetMisses:
    """ancient_dragon has AC 22. Even player_1 (STR 18, prof +3, bonus = +7)
    needs a 15 on the d20 to hit (15+7=22). Roll of 1 → total 8 → miss."""

    def test_is_hit_false(self):
        result = attack("player_1", "ancient_dragon", "longsword", d20_result=1)
        assert result["is_hit"] is False

    def test_damage_zero_on_miss(self):
        result = attack("player_1", "ancient_dragon", "longsword", d20_result=1)
        assert result["damage"] == 0

    def test_target_hp_unchanged_on_miss(self):
        st = fresh_state()
        before = st["npcs"]["ancient_dragon"]["hp"]
        with patch("combat._roll_d20", return_value=1):
            result = resolve_melee_attack("player_1", "ancient_dragon", "longsword", _state=st)
        assert result["target_hp"] == before

    def test_roll_and_total_fields(self):
        result = attack("player_1", "ancient_dragon", "longsword", d20_result=1)
        # player_1: STR 18 (+4), prof +3 → bonus = 7
        assert result["roll"] == 1
        assert result["hit_bonus"] == 7   # +4 STR mod + +3 prof
        assert result["total"] == 8       # 1 + 7
        assert result["ac"] == 22


# ---------------------------------------------------------------------------
# 2. Low-AC target must register a hit on a high roll
# ---------------------------------------------------------------------------

class TestLowAcTargetHits:
    """goblin_1 has AC 12. player_1 with bonus +7: any roll >= 5 hits.
    Roll of 15 → total 22 → definite hit."""

    def test_is_hit_true(self):
        result = attack("player_1", "goblin_1", "longsword", d20_result=15)
        assert result["is_hit"] is True

    def test_damage_positive_on_hit(self):
        result = attack("player_1", "goblin_1", "longsword", d20_result=15)
        assert isinstance(result["damage"], int)
        assert result["damage"] >= 1

    def test_target_hp_reduced(self):
        st = fresh_state()
        before = st["npcs"]["goblin_1"]["hp"]
        with patch("combat._roll_d20", return_value=15):
            result = resolve_melee_attack("player_1", "goblin_1", "longsword", _state=st)
        # HP cannot go below 0 — use max(0, ...) to match engine behaviour
        assert result["target_hp"] == max(0, before - result["damage"])
        assert result["target_hp"] >= 0

    def test_target_hp_cannot_go_below_zero(self):
        """A single hit can drop the goblin to exactly 0, never negative."""
        st = fresh_state()
        # Force maximum longsword damage: 1d8+4 = 12. Goblin has 7 HP.
        with patch("combat._roll_d20", return_value=19), \
             patch("combat.random.randint", side_effect=[8]):   # 1d8 = 8 → 8+4 = 12 > 7
            result = resolve_melee_attack("player_1", "goblin_1", "longsword", _state=st)
        assert result["target_hp"] == 0


# ---------------------------------------------------------------------------
# 3. Return schema — all fields, correct types
# ---------------------------------------------------------------------------

class TestReturnSchema:
    REQUIRED_KEYS = {"roll", "hit_bonus", "total", "ac", "is_hit", "damage",
                     "target_hp", "attacker", "target"}

    def test_all_keys_present(self):
        result = attack("player_1", "goblin_1", "shortsword", d20_result=10)
        assert self.REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - set(result.keys())}"
        )

    def test_int_fields(self):
        result = attack("player_1", "goblin_1", "shortsword", d20_result=10)
        for key in ("roll", "hit_bonus", "total", "ac", "damage", "target_hp"):
            assert isinstance(result[key], int), f"'{key}' should be int, got {type(result[key])}"

    def test_is_hit_is_bool(self):
        result = attack("player_1", "goblin_1", "shortsword", d20_result=10)
        assert isinstance(result["is_hit"], bool)

    def test_string_name_fields(self):
        result = attack("player_1", "goblin_1", "shortsword", d20_result=10)
        assert isinstance(result["attacker"], str) and len(result["attacker"]) > 0
        assert isinstance(result["target"], str) and len(result["target"]) > 0

    def test_roll_in_range(self):
        result = attack("player_1", "goblin_1", "shortsword", d20_result=13)
        assert 1 <= result["roll"] <= 20


# ---------------------------------------------------------------------------
# 4. Input validation — bad inputs must raise, not silently succeed
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_unknown_attacker_raises(self):
        with pytest.raises(KeyError, match="not found"):
            resolve_melee_attack("nobody", "goblin_1", "longsword")

    def test_unknown_target_raises(self):
        with pytest.raises(KeyError, match="not found"):
            resolve_melee_attack("player_1", "phantom_enemy", "longsword")

    def test_unknown_weapon_raises(self):
        with pytest.raises(KeyError, match="Unknown weapon"):
            resolve_melee_attack("player_1", "goblin_1", "bazooka")

    def test_same_entity_raises(self):
        with pytest.raises(ValueError, match="different"):
            resolve_melee_attack("player_1", "player_1", "longsword")

    def test_weapon_slug_normalisation(self):
        """'Long Sword' (spaces, mixed case) should resolve to 'longsword'."""
        result = attack("player_1", "goblin_1", "Long Sword", d20_result=15)
        assert result["is_hit"] is True

    def test_npc_can_attack_player(self):
        """NPCs are valid attackers, not just targets."""
        result = attack("orc_1", "player_2", "handaxe", d20_result=10)
        assert "is_hit" in result


# ---------------------------------------------------------------------------
# 5. STR modifier math verification
# ---------------------------------------------------------------------------

class TestStrModifierMath:
    @pytest.mark.parametrize("score, expected_mod", [
        (10, 0),
        (11, 0),
        (12, 1),
        (13, 1),
        (18, 4),
        (20, 5),
        (8,  -1),
        (7,  -2),
    ])
    def test_str_modifier_formula(self, score: int, expected_mod: int):
        assert _str_modifier(score) == expected_mod

    def test_hit_bonus_composition(self):
        """player_1: STR 18 (+4), prof +3 → hit_bonus must be 7."""
        result = attack("player_1", "goblin_1", "longsword", d20_result=5)
        assert result["hit_bonus"] == 7

    def test_orc_hit_bonus(self):
        """orc_1: STR 16 (+3), prof +2 → hit_bonus must be 5."""
        result = attack("orc_1", "player_1", "handaxe", d20_result=5)
        assert result["hit_bonus"] == 5


# ---------------------------------------------------------------------------
# 6. HP persistence — damage accumulates across calls on the same state
# ---------------------------------------------------------------------------

class TestHpPersistence:
    def test_hp_decreases_across_two_hits(self):
        st = fresh_state()
        before = st["npcs"]["orc_1"]["hp"]

        with patch("combat._roll_d20", return_value=15):
            r1 = resolve_melee_attack("player_1", "orc_1", "shortsword", _state=st)
        hp_after_first = st["npcs"]["orc_1"]["hp"]

        with patch("combat._roll_d20", return_value=15):
            r2 = resolve_melee_attack("player_1", "orc_1", "shortsword", _state=st)
        hp_after_second = st["npcs"]["orc_1"]["hp"]

        assert hp_after_first < before
        assert hp_after_second <= hp_after_first  # could be 0 if one-shotted

    def test_miss_does_not_change_persistent_hp(self):
        st = fresh_state()
        before = st["npcs"]["orc_1"]["hp"]
        with patch("combat._roll_d20", return_value=1):
            resolve_melee_attack("player_1", "orc_1", "shortsword", _state=st)
        assert st["npcs"]["orc_1"]["hp"] == before


# ---------------------------------------------------------------------------
# 7. Natural 20 — always a hit regardless of AC
# ---------------------------------------------------------------------------

class TestNaturalTwenty:
    def test_nat_20_hits_ancient_dragon(self):
        """Even the dragon (AC 22) is hit by a natural 20 + bonus (= 27)."""
        result = attack("player_1", "ancient_dragon", "greataxe", d20_result=20)
        # 20 + 7 (bonus) = 27 >= 22 → hit
        assert result["is_hit"] is True
        assert result["total"] == 27

    def test_nat_20_deals_damage(self):
        result = attack("player_1", "ancient_dragon", "greataxe", d20_result=20)
        assert result["damage"] >= 1


# ---------------------------------------------------------------------------
# 8. Weapon variety — different weapons produce valid integer damage
# ---------------------------------------------------------------------------

class TestWeaponVariety:
    @pytest.mark.parametrize("weapon", [
        "shortsword",
        "greataxe",
        "greatsword",
        "dagger",
        "maul",
        "unarmed",
    ])
    def test_weapon_produces_valid_observation(self, weapon: str):
        result = attack("player_1", "goblin_1", weapon, d20_result=15)
        assert result["is_hit"] is True
        assert isinstance(result["damage"], int)
        assert result["damage"] >= 1
