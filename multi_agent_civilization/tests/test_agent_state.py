import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.agent_state import (
    Needs, Inventory, Relationship, Emotion, EmotionType, Belief,
    Agent, Archetype, PersonalityTraits, ARCHETYPE_PRESETS
)
from models.world_state import ResourceType


class TestNeeds:
    def test_tick_increases_hunger(self):
        n = Needs()
        n.tick()
        assert abs(n.hunger - 0.08) < 0.001

    def test_tick_increases_thirst(self):
        n = Needs()
        n.tick()
        assert abs(n.thirst - 0.10) < 0.001

    def test_tick_increases_rest(self):
        n = Needs()
        n.tick()
        assert abs(n.rest - 0.05) < 0.001

    def test_desperate_detection(self):
        n = Needs()
        n.hunger = 0.90
        assert n.is_desperate

    def test_comfortable_detection(self):
        n = Needs()
        assert not n.is_desperate

    def test_priority_ordering(self):
        n = Needs()
        n.thirst = 0.9
        n.hunger = 0.5
        order = n.priority_order()
        assert order[0][0] == 'thirst'

    def test_auto_consume_food(self):
        n = Needs()
        n.hunger = 0.6
        inv = Inventory(food=5.0)
        n.auto_consume(inv)
        assert inv.food < 5.0

    def test_hunger_floor_at_zero(self):
        n = Needs()
        n.hunger = 0.0
        n.tick()
        assert n.hunger >= 0.0


class TestInventory:
    def test_capacity_limit(self):
        inv = Inventory(max_capacity=20.0)
        added = inv.add(ResourceType.FOOD, 100.0)
        assert inv.food <= 20.0

    def test_remove_actual_amount(self):
        inv = Inventory(food=3.0)
        removed = inv.remove(ResourceType.FOOD, 10.0)
        assert removed == 3.0
        assert inv.food == 0.0

    def test_remaining_capacity(self):
        inv = Inventory(food=5.0, max_capacity=20.0)
        assert abs(inv.remaining_capacity - 15.0) < 0.01


class TestRelationship:
    def test_cooperation_increases_trust(self):
        r = Relationship(trust=0.5)
        r.update_after_cooperation()
        assert r.trust > 0.5

    def test_betrayal_scales_with_prior_trust(self):
        r1 = Relationship(trust=0.9)
        r2 = Relationship(trust=0.2)
        r1_trust_before = r1.trust
        r2_trust_before = r2.trust
        r1.update_after_betrayal()
        r2.update_after_betrayal()
        drop1 = r1_trust_before - r1.trust
        drop2 = r2_trust_before - r2.trust
        assert drop1 > drop2

    def test_decay_toward_neutral(self):
        r = Relationship(trust=0.9, affinity=0.8)
        r.decay()
        assert r.trust < 0.9

    def test_cooperation_ratio(self):
        r = Relationship()
        r.update_after_cooperation()
        r.update_after_cooperation()
        assert r.trust > 0.5


class TestEmotion:
    def test_decay(self):
        e = Emotion(EmotionType.FEAR, intensity=0.5, source="test")
        e.decay()
        assert e.intensity < 0.5

    def test_expiry(self):
        e = Emotion(EmotionType.FEAR, intensity=0.05, source="test", decay_rate=0.1)
        expired = e.decay()
        assert expired


class TestBelief:
    def test_reinforce(self):
        b = Belief(content="test", confidence=0.5, source="test")
        b.reinforce()
        assert b.confidence > 0.5

    def test_contradict(self):
        b = Belief(content="test", confidence=0.5, source="test")
        b.contradict()
        assert b.confidence < 0.5

    def test_superstition_detection(self):
        b = Belief(content="magic", confidence=0.8, source="", falsifiability=0.1)
        assert b.is_superstition


class TestAgent:
    def _make_agent(self, archetype=Archetype.BUILDER):
        preset = ARCHETYPE_PRESETS[archetype].copy()
        valid_fields = PersonalityTraits.__dataclass_fields__.keys()
        personality = PersonalityTraits(**{k: v for k, v in preset.items() if k in valid_fields})
        return Agent(
            id="test_agent",
            name="Test",
            archetype=archetype,
            personality=personality,
        )

    def test_cognitive_bandwidth_under_stress(self):
        agent = self._make_agent()
        agent.needs.hunger = 0.9
        agent.needs.thirst = 0.9
        cb = agent.cognitive_bandwidth
        assert cb >= 1

    def test_memory_pruning(self):
        agent = self._make_agent()
        from models.agent_state import MemoryEntry
        for i in range(205):
            agent.add_memory(MemoryEntry(turn=i, category="test", content=f"memory {i}", importance=0.5))
        assert len(agent.memories) <= 200

    def test_emotion_stacking(self):
        agent = self._make_agent()
        agent.emotions.append(Emotion(EmotionType.FEAR, intensity=0.8, source="test"))
        agent.emotions.append(Emotion(EmotionType.ANGER, intensity=0.5, source="test"))
        assert agent.dominant_emotion.emotion_type == EmotionType.FEAR


class TestWorldInit:
    def test_all_archetypes_have_presets(self):
        for archetype in Archetype:
            assert archetype in ARCHETYPE_PRESETS

    def test_personality_values_in_range(self):
        for archetype, preset in ARCHETYPE_PRESETS.items():
            for key, val in preset.items():
                assert 0.0 <= val <= 1.0, f"{archetype}.{key} = {val} out of range"
