"""
agent_state.py — Core agent data structures for the Advanced Civilization simulation.

Defines the Agent dataclass along with supporting structures for emotions, mood,
memory, and complex emotions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Emotion types
# ---------------------------------------------------------------------------

class EmotionType(Enum):
    """The 9 base emotions an agent can experience."""
    FEAR = "fear"
    ANGER = "anger"
    HOPE = "hope"
    GRIEF = "grief"
    GRATITUDE = "gratitude"
    PRIDE = "pride"
    ENVY = "envy"
    LONELINESS = "loneliness"
    DISGUST = "disgust"


# Contagion multipliers per emotion (how easily an emotion spreads)
CONTAGION_MULTIPLIERS: Dict[EmotionType, float] = {
    EmotionType.FEAR: 1.0,
    EmotionType.ANGER: 1.0,
    EmotionType.HOPE: 0.7,
    EmotionType.GRATITUDE: 0.7,
    EmotionType.GRIEF: 0.7,
    EmotionType.PRIDE: 0.3,
    EmotionType.ENVY: 0.3,
    EmotionType.LONELINESS: 0.5,
    EmotionType.DISGUST: 0.6,
}


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------

@dataclass
class Emotion:
    """A single emotion held by an agent."""
    emotion_type: EmotionType
    intensity: float = 0.0          # 0.0–1.0
    decay_rate: float = 0.05        # Fraction lost per turn
    turns_active: int = 0           # How many turns this emotion has been active

    def tick_decay(self) -> None:
        """Reduce intensity by decay_rate each turn."""
        self.intensity = max(0.0, self.intensity - self.decay_rate)
        if self.intensity > 0:
            self.turns_active += 1
        else:
            self.turns_active = 0

    def add_intensity(self, amount: float) -> None:
        """Stack additional intensity (clamped to 1.0)."""
        self.intensity = min(1.0, self.intensity + amount)


# ---------------------------------------------------------------------------
# Mood
# ---------------------------------------------------------------------------

@dataclass
class Mood:
    """Slow-moving emotional baseline that shifts over many turns."""
    optimism: float = 0.0           # -1.0 (pessimistic) to 1.0 (optimistic)
    stability: float = 0.5          # 0.0 (volatile) to 1.0 (stable)
    sociability: float = 0.5        # 0.0 (withdrawn) to 1.0 (outgoing)
    aggression_baseline: float = 0.0  # -0.5 (pacifist) to 0.5 (aggressive)

    # Max change per turn
    MAX_OPTIMISM_DELTA: float = field(default=0.03, init=False, repr=False)
    MAX_STABILITY_DELTA: float = field(default=0.02, init=False, repr=False)
    MAX_SOCIABILITY_DELTA: float = field(default=0.02, init=False, repr=False)

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def update(
        self,
        recent_emotions: List[Emotion],
        needs_satisfied: bool,
        social_cooperation: bool,
        social_betrayal: bool,
        health: float,
    ) -> None:
        """Update mood based on recent emotional history and environmental factors."""
        # --- Optimism ---
        positive = {EmotionType.HOPE, EmotionType.GRATITUDE, EmotionType.PRIDE}
        negative = {EmotionType.FEAR, EmotionType.ANGER, EmotionType.GRIEF,
                    EmotionType.LONELINESS, EmotionType.ENVY, EmotionType.DISGUST}

        opt_delta = 0.0
        for emo in recent_emotions:
            if emo.emotion_type in positive and emo.intensity > 0.1:
                opt_delta += 0.01
            elif emo.emotion_type in negative and emo.intensity > 0.1:
                opt_delta -= 0.01

        if needs_satisfied:
            opt_delta += 0.01
        else:
            opt_delta -= 0.01

        if health < 0.3:
            opt_delta -= 0.01

        opt_delta = self._clamp(opt_delta, -self.MAX_OPTIMISM_DELTA, self.MAX_OPTIMISM_DELTA)
        self.optimism = self._clamp(self.optimism + opt_delta, -1.0, 1.0)

        # --- Stability ---
        stab_delta = 0.0
        if needs_satisfied:
            stab_delta += 0.01
        else:
            stab_delta -= 0.01
        stab_delta = self._clamp(stab_delta, -self.MAX_STABILITY_DELTA, self.MAX_STABILITY_DELTA)
        self.stability = self._clamp(self.stability + stab_delta, 0.0, 1.0)

        # --- Sociability ---
        soc_delta = 0.0
        if social_cooperation:
            soc_delta += 0.01
        if social_betrayal:
            soc_delta -= 0.02
        soc_delta = self._clamp(soc_delta, -self.MAX_SOCIABILITY_DELTA, self.MAX_SOCIABILITY_DELTA)
        self.sociability = self._clamp(self.sociability + soc_delta, 0.0, 1.0)

    def optimism_description(self) -> str:
        if self.optimism > 0.5:
            return "highly optimistic"
        if self.optimism > 0.1:
            return "cautiously hopeful"
        if self.optimism > -0.1:
            return "neutral"
        if self.optimism > -0.5:
            return "somewhat pessimistic"
        return "deeply pessimistic"

    def stability_description(self) -> str:
        if self.stability > 0.7:
            return "emotionally stable"
        if self.stability > 0.4:
            return "moderately stable"
        return "emotionally volatile"

    def sociability_description(self) -> str:
        if self.sociability > 0.7:
            return "outgoing and social"
        if self.sociability > 0.4:
            return "moderately social"
        return "withdrawn and reserved"

    def prompt_description(self) -> str:
        return (
            f"Your underlying mood: {self.optimism_description()}, "
            f"{self.stability_description()}, {self.sociability_description()}."
        )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory held by an agent."""
    content: str
    importance: float = 0.5            # 0.0–1.0
    turn_created: int = 0
    emotional_intensity_at_creation: float = 0.0  # dominant emotion intensity when formed
    tags: List[str] = field(default_factory=list)

    def is_vivid(self) -> bool:
        return self.emotional_intensity_at_creation > 0.7


# ---------------------------------------------------------------------------
# Complex emotions
# ---------------------------------------------------------------------------

@dataclass
class ComplexEmotion:
    """A composite emotion derived from combinations of base emotions."""
    name: str
    component_emotions: List[Tuple[EmotionType, float]]  # (emotion_type, min_intensity)
    intensity: float = 0.0
    behavioral_effect: str = ""

    def prompt_description(self) -> str:
        return f"{self.name.capitalize()} (intensity {self.intensity:.2f}): {self.behavioral_effect}"


# ---------------------------------------------------------------------------
# Personality traits (Big Five-inspired)
# ---------------------------------------------------------------------------

@dataclass
class Personality:
    extraversion: float = 0.5    # 0.0–1.0
    neuroticism: float = 0.5     # 0.0–1.0
    conscientiousness: float = 0.5  # 0.0–1.0
    honesty: float = 0.5         # 0.0–1.0 (willingness to express true emotions)
    agreeableness: float = 0.5   # 0.0–1.0


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    agent_id: str
    trust: float = 0.5           # 0.0–1.0
    affinity: float = 0.5        # 0.0–1.0
    interaction_count: int = 0
    respect: float = 0.5         # 0.0–1.0


# ---------------------------------------------------------------------------
# Needs
# ---------------------------------------------------------------------------

@dataclass
class Needs:
    safety: float = 0.5          # 0.0 (desperate) to 1.0 (fully satisfied)
    belonging: float = 0.5
    esteem: float = 0.5
    sustenance: float = 0.5      # Food/water
    health: float = 1.0

    def is_satisfied(self) -> bool:
        return (
            self.safety > 0.4
            and self.belonging > 0.3
            and self.sustenance > 0.4
            and self.health > 0.4
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Core agent with emotional, memory, and social state."""
    agent_id: str
    name: str
    region: str = "default"

    # Personality
    personality: Personality = field(default_factory=Personality)

    # Needs
    needs: Needs = field(default_factory=Needs)

    # Base emotions — one slot per EmotionType
    emotions: Dict[EmotionType, Emotion] = field(default_factory=dict)

    # Mood
    mood: Mood = field(default_factory=Mood)

    # Complex emotions detected this turn
    complex_emotions: List[ComplexEmotion] = field(default_factory=list)

    # Memories
    memories: List[MemoryEntry] = field(default_factory=list)

    # Relationships keyed by agent_id
    relationships: Dict[str, Relationship] = field(default_factory=dict)

    # Breaking point tracking — (breaking_point_type, turn)
    breaking_point_history: List[Tuple[str, int]] = field(default_factory=list)

    # Temporary behavioral flags (set by breaking points, reset after duration)
    refuses_trade_turns: int = 0       # Remaining turns of trade refusal
    reduced_action_turns: int = 0      # Remaining turns of reduced actions
    silent_turns: int = 0              # Remaining turns of strategic silence
    aggression_boost_turns: int = 0    # Remaining turns of boosted aggression
    aggression_boost: float = 0.0      # Extra aggression from rage breaking point

    # Obsession target (from envy breaking point)
    obsession_target: Optional[str] = None

    # Inventory / resources (simple key-value)
    inventory: Dict[str, float] = field(default_factory=dict)

    # Action points per turn
    action_points: int = 3

    def __post_init__(self) -> None:
        # Ensure an Emotion slot exists for every EmotionType
        for et in EmotionType:
            if et not in self.emotions:
                self.emotions[et] = Emotion(emotion_type=et)

    # ------------------------------------------------------------------
    # Emotion helpers
    # ------------------------------------------------------------------

    def get_emotion(self, emotion_type: EmotionType) -> Emotion:
        return self.emotions[emotion_type]

    def set_emotion(self, emotion_type: EmotionType, intensity: float) -> None:
        self.emotions[emotion_type].intensity = max(0.0, min(1.0, intensity))

    def add_emotion(self, emotion_type: EmotionType, amount: float) -> None:
        self.emotions[emotion_type].add_intensity(amount)

    def dominant_emotion(self) -> Optional[Emotion]:
        """Return the emotion with the highest current intensity, or None."""
        active = [e for e in self.emotions.values() if e.intensity > 0]
        if not active:
            return None
        return max(active, key=lambda e: e.intensity)

    def dominant_emotion_intensity(self) -> float:
        dom = self.dominant_emotion()
        return dom.intensity if dom else 0.0

    def tick_emotions(self) -> None:
        """Decay all emotions by one turn and track durations."""
        for emotion in self.emotions.values():
            emotion.tick_decay()
        # Decrement temporary flag counters
        self.refuses_trade_turns = max(0, self.refuses_trade_turns - 1)
        self.reduced_action_turns = max(0, self.reduced_action_turns - 1)
        self.silent_turns = max(0, self.silent_turns - 1)
        if self.aggression_boost_turns > 0:
            self.aggression_boost_turns -= 1
            if self.aggression_boost_turns == 0:
                self.aggression_boost = 0.0

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        turn: int = 0,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """Add a memory, automatically capturing the current emotional intensity."""
        emotional_intensity = self.dominant_emotion_intensity()
        # Vivid memories get an importance boost
        if emotional_intensity > 0.7:
            importance = min(1.0, importance + 0.15)
        entry = MemoryEntry(
            content=content,
            importance=importance,
            turn_created=turn,
            emotional_intensity_at_creation=emotional_intensity,
            tags=tags or [],
        )
        self.memories.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------------

    def get_or_create_relationship(self, agent_id: str) -> Relationship:
        if agent_id not in self.relationships:
            self.relationships[agent_id] = Relationship(agent_id=agent_id)
        return self.relationships[agent_id]

    def get_trust(self, agent_id: str) -> float:
        rel = self.relationships.get(agent_id)
        return rel.trust if rel else 0.5

    def lowest_affinity_agent(self) -> Optional[str]:
        if not self.relationships:
            return None
        return min(self.relationships.values(), key=lambda r: r.affinity).agent_id

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def emotion_summary(self) -> str:
        """Human-readable summary of active emotions for LLM prompts."""
        active = [(et, e) for et, e in self.emotions.items() if e.intensity > 0.05]
        if not active:
            return "You feel emotionally neutral."
        parts = [f"{et.value} ({e.intensity:.2f})" for et, e in active]
        return "Current emotions: " + ", ".join(parts) + "."

    def complex_emotion_summary(self) -> str:
        if not self.complex_emotions:
            return ""
        descs = [ce.prompt_description() for ce in self.complex_emotions]
        return "You are also experiencing complex emotions: " + "; ".join(descs) + "."

    def speech_style_prompt(self) -> str:
        return (
            "Your emotional state strongly colors how you speak. When angry, you are blunt "
            "and confrontational. When fearful, you are hesitant and seek reassurance. When "
            "grateful, you are warm and generous with words. When grieving, you are quiet and "
            "somber. Let your emotions show in your words — or deliberately hide them if your "
            "personality (honesty trait) allows deception."
        )
