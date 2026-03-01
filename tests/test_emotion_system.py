"""
tests/test_emotion_system.py — Tests for the Advanced Civilization emotion system.

Covers:
- TestEmotionalContagion
- TestMood
- TestEmotionalMemory
- TestBreakingPoints
- TestComplexEmotions
"""

from __future__ import annotations

import random
import sys
import os

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent_state import (
    Agent,
    Emotion,
    EmotionType,
    MemoryEntry,
    Mood,
    Personality,
    Needs,
)
from emotion_system import (
    EmotionalContagion,
    EmotionalBreakingPoint,
    ComplexEmotionDetector,
    CATHARSIS_INTENSITY,
)
from memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(
    agent_id: str = "agent_1",
    name: str = "Alice",
    region: str = "north",
    extraversion: float = 0.5,
    neuroticism: float = 0.5,
    conscientiousness: float = 0.5,
    honesty: float = 0.7,
) -> Agent:
    agent = Agent(
        agent_id=agent_id,
        name=name,
        region=region,
        personality=Personality(
            extraversion=extraversion,
            neuroticism=neuroticism,
            conscientiousness=conscientiousness,
            honesty=honesty,
        ),
    )
    return agent


# ---------------------------------------------------------------------------
# TestEmotionalContagion
# ---------------------------------------------------------------------------

class TestEmotionalContagion:
    """Fear (high contagion) should spread; low-contagion emotions spread less easily."""

    def _run_contagion(
        self,
        source: Agent,
        recipient: Agent,
        iterations: int = 100,
        rng_seed: int = 42,
    ) -> int:
        """Run contagion many times and count how often recipient catches an emotion."""
        contagion = EmotionalContagion()
        caught_count = 0
        rng = random.Random(rng_seed)
        for _ in range(iterations):
            # Reset recipient's emotions each iteration
            for et in EmotionType:
                recipient.emotions[et].intensity = 0.0
            contagion.run_pass([source, recipient], turn=1, rng=rng)
            if recipient.get_emotion(EmotionType.FEAR).intensity > 0:
                caught_count += 1
        return caught_count

    def test_fear_spreads_between_agents(self) -> None:
        """Fear should spread from source to recipient in the same region."""
        source = make_agent("src", "Source", region="zone_a", extraversion=0.8)
        source.set_emotion(EmotionType.FEAR, 0.9)

        recipient = make_agent("rec", "Recipient", region="zone_a", neuroticism=0.8)

        contagion = EmotionalContagion()
        rng = random.Random(0)

        # Run many trials — fear should spread at least some of the time
        spread_count = 0
        for _ in range(50):
            recipient.emotions[EmotionType.FEAR].intensity = 0.0
            contagion.run_pass([source, recipient], turn=1, rng=rng)
            if recipient.get_emotion(EmotionType.FEAR).intensity > 0:
                spread_count += 1

        assert spread_count > 0, "Fear should spread at least sometimes"

    def test_low_contagion_emotion_spreads_less(self) -> None:
        """Pride (contagion multiplier 0.3) should spread less than fear (1.0)."""
        def count_spreads(emotion_type: EmotionType, trials: int = 200) -> int:
            contagion = EmotionalContagion()
            rng = random.Random(99)
            count = 0
            for _ in range(trials):
                source = make_agent("src", "Source", region="zone_b", extraversion=0.9)
                source.set_emotion(emotion_type, 0.9)
                recipient = make_agent("rec", "Recipient", region="zone_b", neuroticism=0.5)
                contagion.run_pass([source, recipient], turn=1, rng=rng)
                if recipient.get_emotion(emotion_type).intensity > 0:
                    count += 1
            return count

        fear_count = count_spreads(EmotionType.FEAR)
        pride_count = count_spreads(EmotionType.PRIDE)
        assert fear_count > pride_count, (
            f"Fear ({fear_count}) should spread more than pride ({pride_count})"
        )

    def test_neuroticism_increases_susceptibility(self) -> None:
        """High-neuroticism agent should catch contagion more often than low-neuroticism."""
        contagion = EmotionalContagion()

        def count_catches(neuroticism: float, trials: int = 200) -> int:
            rng = random.Random(7)
            count = 0
            for _ in range(trials):
                source = make_agent("src", "Src", region="z", extraversion=0.8)
                source.set_emotion(EmotionType.FEAR, 0.8)
                recipient = make_agent("rec", "Rec", region="z", neuroticism=neuroticism)
                contagion.run_pass([source, recipient], turn=1, rng=rng)
                if recipient.get_emotion(EmotionType.FEAR).intensity > 0:
                    count += 1
            return count

        high_neuro = count_catches(0.9)
        low_neuro = count_catches(0.1)
        assert high_neuro > low_neuro, (
            f"High neuroticism ({high_neuro}) should catch more than low ({low_neuro})"
        )

    def test_different_regions_no_contagion(self) -> None:
        """Agents in different regions should not experience contagion."""
        source = make_agent("src", "Source", region="region_a")
        source.set_emotion(EmotionType.FEAR, 1.0)

        recipient = make_agent("rec", "Recipient", region="region_b")

        contagion = EmotionalContagion()
        rng = random.Random(0)

        for _ in range(50):
            recipient.emotions[EmotionType.FEAR].intensity = 0.0
            contagion.run_pass([source, recipient], turn=1, rng=rng)

        assert recipient.get_emotion(EmotionType.FEAR).intensity == 0.0, (
            "Agents in different regions should not share emotions"
        )

    def test_contagion_memory_added(self) -> None:
        """Recipient should gain a memory entry when catching an emotion."""
        source = make_agent("src", "Bobo", region="z", extraversion=1.0)
        source.set_emotion(EmotionType.FEAR, 1.0)

        # Use fixed RNG to guarantee contagion happens
        contagion = EmotionalContagion()
        rng = random.Random(0)

        caught = False
        for _ in range(100):
            recipient = make_agent("rec", "Rec", region="z", neuroticism=1.0)
            contagion.run_pass([source, recipient], turn=1, rng=rng)
            if recipient.get_emotion(EmotionType.FEAR).intensity > 0:
                assert any("spreading from" in m.content for m in recipient.memories)
                caught = True
                break

        assert caught, "Should catch emotion at least once in 100 trials"


# ---------------------------------------------------------------------------
# TestMood
# ---------------------------------------------------------------------------

class TestMood:
    """Mood updates from emotion history, changes are capped, desperation trends pessimistic."""

    def test_positive_emotions_increase_optimism(self) -> None:
        mood = Mood()
        agent = make_agent()
        agent.set_emotion(EmotionType.HOPE, 0.8)
        agent.set_emotion(EmotionType.GRATITUDE, 0.6)
        recent = list(agent.emotions.values())

        initial_optimism = mood.optimism
        mood.update(
            recent_emotions=recent,
            needs_satisfied=True,
            social_cooperation=False,
            social_betrayal=False,
            health=1.0,
        )
        assert mood.optimism > initial_optimism, "Positive emotions should increase optimism"

    def test_negative_emotions_decrease_optimism(self) -> None:
        mood = Mood()
        agent = make_agent()
        agent.set_emotion(EmotionType.FEAR, 0.9)
        agent.set_emotion(EmotionType.GRIEF, 0.8)
        recent = list(agent.emotions.values())

        initial_optimism = mood.optimism
        mood.update(
            recent_emotions=recent,
            needs_satisfied=False,
            social_cooperation=False,
            social_betrayal=False,
            health=0.2,
        )
        assert mood.optimism < initial_optimism, "Negative emotions should decrease optimism"

    def test_mood_changes_capped_at_max_rate(self) -> None:
        """A single update should never move optimism by more than MAX_OPTIMISM_DELTA."""
        mood = Mood()
        agent = make_agent()
        # Load up positive emotions maximally
        for et in [EmotionType.HOPE, EmotionType.GRATITUDE, EmotionType.PRIDE]:
            agent.set_emotion(et, 1.0)
        recent = list(agent.emotions.values())

        for _ in range(5):
            before = mood.optimism
            mood.update(
                recent_emotions=recent,
                needs_satisfied=True,
                social_cooperation=True,
                social_betrayal=False,
                health=1.0,
            )
            delta = abs(mood.optimism - before)
            assert delta <= mood.MAX_OPTIMISM_DELTA + 1e-9, (
                f"Optimism delta {delta} exceeds max {mood.MAX_OPTIMISM_DELTA}"
            )

    def test_desperate_agents_trend_pessimistic(self) -> None:
        """Agents with unmet needs and negative emotions should become more pessimistic over time."""
        mood = Mood(optimism=0.5)
        agent = make_agent()
        agent.set_emotion(EmotionType.FEAR, 0.9)
        agent.set_emotion(EmotionType.LONELINESS, 0.8)
        recent = list(agent.emotions.values())

        for _ in range(20):
            mood.update(
                recent_emotions=recent,
                needs_satisfied=False,
                social_cooperation=False,
                social_betrayal=True,
                health=0.1,
            )

        assert mood.optimism < 0.5, "Desperate conditions should decrease optimism over time"

    def test_optimism_clamped_to_range(self) -> None:
        """Optimism must stay within [-1.0, 1.0]."""
        mood = Mood(optimism=0.99)
        agent = make_agent()
        agent.set_emotion(EmotionType.HOPE, 1.0)
        recent = list(agent.emotions.values())

        for _ in range(100):
            mood.update(
                recent_emotions=recent,
                needs_satisfied=True,
                social_cooperation=True,
                social_betrayal=False,
                health=1.0,
            )

        assert mood.optimism <= 1.0, "Optimism must not exceed 1.0"

    def test_mood_stability_decreases_when_desperate(self) -> None:
        mood = Mood(stability=0.8)
        agent = make_agent()
        recent = list(agent.emotions.values())

        for _ in range(10):
            mood.update(
                recent_emotions=recent,
                needs_satisfied=False,
                social_cooperation=False,
                social_betrayal=False,
                health=1.0,
            )

        assert mood.stability < 0.8, "Stability should decrease when needs are unmet"


# ---------------------------------------------------------------------------
# TestEmotionalMemory
# ---------------------------------------------------------------------------

class TestEmotionalMemory:
    """High-emotion memories get recall bonus; vivid memories resist pruning."""

    def test_high_emotion_memory_recall_bonus(self) -> None:
        """Memories formed during high-emotion states score higher in retrieval."""
        memories = [
            MemoryEntry(
                content="calm event",
                importance=0.5,
                turn_created=0,
                emotional_intensity_at_creation=0.1,
            ),
            MemoryEntry(
                content="terrifying event",
                importance=0.5,
                turn_created=0,
                emotional_intensity_at_creation=0.9,
            ),
        ]
        mm = MemoryManager(memories)
        retrieved = mm.retrieve(current_turn=0, top_n=2)
        # The high-emotion memory should rank first
        assert retrieved[0].content == "terrifying event", (
            "High-emotion memory should rank above low-emotion memory of equal base importance"
        )

    def test_vivid_memories_in_context_summary(self) -> None:
        """Vivid memories (emotional_intensity > 0.7) appear in EMOTIONALLY VIVID MEMORIES section."""
        memories = [
            MemoryEntry("ordinary", importance=0.5, turn_created=0, emotional_intensity_at_creation=0.2),
            MemoryEntry("traumatic", importance=0.5, turn_created=0, emotional_intensity_at_creation=0.9),
        ]
        mm = MemoryManager(memories)
        summary = mm.build_context_summary(current_turn=0)
        assert "EMOTIONALLY VIVID MEMORIES" in summary
        assert "traumatic" in summary

    def test_vivid_memories_resist_pruning(self) -> None:
        """Vivid memories should survive pruning more often than low-emotion memories."""
        mm_size = MemoryManager.MAX_MEMORIES + 10
        memories = []
        for i in range(mm_size):
            # Mix: last 5 are vivid
            intensity = 0.9 if i >= mm_size - 5 else 0.05
            memories.append(
                MemoryEntry(
                    content=f"memory_{i}",
                    importance=0.3,
                    turn_created=i,
                    emotional_intensity_at_creation=intensity,
                )
            )

        mm = MemoryManager(memories)
        mm.prune_if_needed(current_turn=mm_size)

        surviving = {m.content for m in memories}
        for i in range(mm_size - 5, mm_size):
            assert f"memory_{i}" in surviving, (
                f"Vivid memory memory_{i} should survive pruning"
            )

    def test_add_memory_captures_emotion_intensity(self) -> None:
        """Agent.add_memory() should capture the current dominant emotion intensity."""
        agent = make_agent()
        agent.set_emotion(EmotionType.FEAR, 0.8)

        mem = agent.add_memory("Something frightening happened", importance=0.5, turn=1)
        assert abs(mem.emotional_intensity_at_creation - 0.8) < 0.01, (
            "Memory should capture dominant emotion intensity at creation"
        )

    def test_vivid_memory_importance_boosted(self) -> None:
        """Memories formed during high-emotion (>0.7) get importance boosted by 0.15."""
        agent = make_agent()
        agent.set_emotion(EmotionType.GRIEF, 0.85)

        mem = agent.add_memory("Lost everything", importance=0.5, turn=1)
        assert mem.importance >= 0.65 - 1e-9, (
            f"Vivid memory importance should be boosted (got {mem.importance})"
        )

    def test_vivid_memories_top_5_by_intensity(self) -> None:
        """MemoryManager.vivid_memories() returns top 5 by emotional intensity."""
        memories = [
            MemoryEntry(f"mem_{i}", importance=0.5, turn_created=0,
                        emotional_intensity_at_creation=float(i) / 10)
            for i in range(20)
        ]
        mm = MemoryManager(memories)
        vivid = mm.vivid_memories()
        assert len(vivid) == 5
        intensities = [m.emotional_intensity_at_creation for m in vivid]
        assert all(intensities[i] >= intensities[i + 1] for i in range(4)), (
            "Vivid memories should be ordered by intensity descending"
        )


# ---------------------------------------------------------------------------
# TestBreakingPoints
# ---------------------------------------------------------------------------

class TestBreakingPoints:
    """Breaking points trigger after sustained high-intensity emotions."""

    def _setup_agent_with_sustained_emotion(
        self, emotion_type: EmotionType, turns: int, intensity: float = 0.8
    ) -> Agent:
        agent = make_agent()
        emo = agent.get_emotion(emotion_type)
        emo.intensity = intensity
        emo.turns_active = turns
        return agent

    def test_fear_breaking_point_triggers_paranoia_after_5_turns(self) -> None:
        agent = self._setup_agent_with_sustained_emotion(EmotionType.FEAR, turns=5)
        # Add a relationship to test trust drop
        rel = agent.get_or_create_relationship("other_agent")
        rel.trust = 0.8

        bp = EmotionalBreakingPoint()
        events = bp.check_agents([agent], [], turn=1)

        paranoia_events = [e for e in events if e.breaking_point_type == "Paranoia"]
        assert len(paranoia_events) == 1, "Paranoia should trigger after 5 turns of fear > 0.6"
        assert rel.trust < 0.8, "Trust should drop after Paranoia"
        assert agent.refuses_trade_turns > 0, "Agent should refuse trades"

    def test_anger_breaking_point_triggers_rage_after_4_turns(self) -> None:
        agent = self._setup_agent_with_sustained_emotion(EmotionType.ANGER, turns=4)

        bp = EmotionalBreakingPoint()
        events = bp.check_agents([agent], [], turn=1)

        rage_events = [e for e in events if e.breaking_point_type == "Rage"]
        assert len(rage_events) == 1, "Rage should trigger after 4 turns of anger > 0.6"
        assert agent.aggression_boost > 0, "Aggression boost should be set"
        assert agent.aggression_boost_turns > 0

    def test_grief_breaking_point_triggers_depression_after_6_turns(self) -> None:
        agent = self._setup_agent_with_sustained_emotion(EmotionType.GRIEF, turns=6)

        bp = EmotionalBreakingPoint()
        events = bp.check_agents([agent], [], turn=1)

        depression_events = [e for e in events if e.breaking_point_type == "Depression"]
        assert len(depression_events) == 1
        assert agent.reduced_action_turns > 0
        assert agent.silent_turns > 0

    def test_loneliness_breaking_point_triggers_desperation_after_5_turns(self) -> None:
        agent = self._setup_agent_with_sustained_emotion(EmotionType.LONELINESS, turns=5)

        bp = EmotionalBreakingPoint()
        events = bp.check_agents([agent], [], turn=1)

        desp_events = [e for e in events if e.breaking_point_type == "Desperation"]
        assert len(desp_events) == 1
        assert agent.needs.belonging == 1.0

    def test_emotion_resets_after_breaking_point(self) -> None:
        """Emotion should reset to CATHARSIS_INTENSITY (0.3) after a breaking point."""
        agent = self._setup_agent_with_sustained_emotion(EmotionType.FEAR, turns=5)

        bp = EmotionalBreakingPoint()
        bp.check_agents([agent], [], turn=1)

        fear = agent.get_emotion(EmotionType.FEAR)
        assert abs(fear.intensity - CATHARSIS_INTENSITY) < 0.01, (
            f"Fear should reset to {CATHARSIS_INTENSITY} after breaking point, got {fear.intensity}"
        )

    def test_no_breaking_point_if_not_enough_turns(self) -> None:
        """Breaking point should NOT trigger if emotion hasn't been sustained long enough."""
        # Fear needs 5 turns — only 3 here
        agent = self._setup_agent_with_sustained_emotion(EmotionType.FEAR, turns=3)

        bp = EmotionalBreakingPoint()
        events = bp.check_agents([agent], [], turn=1)

        paranoia_events = [e for e in events if e.breaking_point_type == "Paranoia"]
        assert len(paranoia_events) == 0, "Paranoia should NOT trigger with only 3 turns"

    def test_breaking_point_recorded_in_history(self) -> None:
        agent = self._setup_agent_with_sustained_emotion(EmotionType.ANGER, turns=4)

        bp = EmotionalBreakingPoint()
        bp.check_agents([agent], [], turn=5)

        assert len(agent.breaking_point_history) == 1
        bp_type, bp_turn = agent.breaking_point_history[0]
        assert bp_type == "Rage"
        assert bp_turn == 5

    def test_witness_gets_strong_memory(self) -> None:
        """Witnesses of a breaking point should form a high-importance memory."""
        agent = self._setup_agent_with_sustained_emotion(EmotionType.FEAR, turns=5)
        agent.region = "shared_zone"
        witness = make_agent("wit", "Witness", region="shared_zone")

        bp = EmotionalBreakingPoint()
        bp.check_agents([agent], [agent, witness], turn=3)

        high_importance = [m for m in witness.memories if m.importance >= 0.9]
        assert len(high_importance) >= 1, "Witness should have a high-importance memory"


# ---------------------------------------------------------------------------
# TestComplexEmotions
# ---------------------------------------------------------------------------

class TestComplexEmotions:
    """Complex emotions are detected from base emotion combinations and decay with components."""

    def test_jealousy_detected_from_envy_and_fear(self) -> None:
        agent = make_agent()
        agent.set_emotion(EmotionType.ENVY, 0.5)
        agent.set_emotion(EmotionType.FEAR, 0.4)

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        names = [ce.name for ce in complex_emotions]
        assert "jealousy" in names, f"Jealousy should be detected, got: {names}"

    def test_contempt_detected_from_anger_and_disgust(self) -> None:
        agent = make_agent()
        agent.set_emotion(EmotionType.ANGER, 0.5)
        agent.set_emotion(EmotionType.DISGUST, 0.4)

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        names = [ce.name for ce in complex_emotions]
        assert "contempt" in names, f"Contempt should be detected, got: {names}"

    def test_complex_emotion_not_detected_when_components_below_threshold(self) -> None:
        """Jealousy should NOT appear if envy or fear is below the minimum."""
        agent = make_agent()
        agent.set_emotion(EmotionType.ENVY, 0.1)   # Below 0.3
        agent.set_emotion(EmotionType.FEAR, 0.4)

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        names = [ce.name for ce in complex_emotions]
        assert "jealousy" not in names, "Jealousy should NOT be detected with envy below threshold"

    def test_complex_emotions_decay_when_components_decay(self) -> None:
        """Complex emotions should disappear when their component emotions decay below threshold."""
        agent = make_agent()
        agent.set_emotion(EmotionType.ENVY, 0.5)
        agent.set_emotion(EmotionType.FEAR, 0.4)

        detector = ComplexEmotionDetector()
        # Initially detected
        ce = detector.detect(agent)
        assert any(c.name == "jealousy" for c in ce)

        # Decay components below threshold
        agent.set_emotion(EmotionType.ENVY, 0.1)
        agent.set_emotion(EmotionType.FEAR, 0.05)

        ce_after = detector.detect(agent)
        assert not any(c.name == "jealousy" for c in ce_after), (
            "Jealousy should no longer be detected after components decay"
        )

    def test_admiration_detected_from_gratitude_hope_and_respect(self) -> None:
        agent = make_agent()
        agent.set_emotion(EmotionType.GRATITUDE, 0.5)
        agent.set_emotion(EmotionType.HOPE, 0.4)
        # Add a relationship with high respect
        rel = agent.get_or_create_relationship("mentor")
        rel.respect = 0.8

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        names = [ce.name for ce in complex_emotions]
        assert "admiration" in names, f"Admiration should be detected, got: {names}"

    def test_awe_requires_high_respect(self) -> None:
        """Awe should NOT be detected without a relationship with respect > 0.7."""
        agent = make_agent()
        agent.set_emotion(EmotionType.FEAR, 0.4)
        agent.set_emotion(EmotionType.GRATITUDE, 0.4)
        # No relationships → no high-respect relationship

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        names = [ce.name for ce in complex_emotions]
        assert "awe" not in names, "Awe should not be detected without a high-respect relationship"

    def test_awe_detected_with_high_respect(self) -> None:
        agent = make_agent()
        agent.set_emotion(EmotionType.FEAR, 0.4)
        agent.set_emotion(EmotionType.GRATITUDE, 0.4)
        rel = agent.get_or_create_relationship("elder")
        rel.respect = 0.9

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        names = [ce.name for ce in complex_emotions]
        assert "awe" in names, f"Awe should be detected, got: {names}"

    def test_complex_emotion_intensity_is_average_of_components(self) -> None:
        agent = make_agent()
        agent.set_emotion(EmotionType.ENVY, 0.6)
        agent.set_emotion(EmotionType.FEAR, 0.4)

        detector = ComplexEmotionDetector()
        complex_emotions = detector.detect(agent)

        jealousy = next(ce for ce in complex_emotions if ce.name == "jealousy")
        expected = round((0.6 + 0.4) / 2, 3)
        assert abs(jealousy.intensity - expected) < 0.01, (
            f"Jealousy intensity should be average of components: {expected}, got {jealousy.intensity}"
        )
