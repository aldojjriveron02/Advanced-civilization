"""
emotion_system.py — Advanced emotional processing systems.

Includes:
- EmotionalContagion: emotions spread between co-located agents.
- EmotionalBreakingPoint: prolonged high-intensity emotions cause dramatic shifts.
- ComplexEmotionDetector: detects composite emotions from base emotion combinations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agent_state import (
    Agent,
    ComplexEmotion,
    EmotionType,
    CONTAGION_MULTIPLIERS,
)


# ---------------------------------------------------------------------------
# Emotional Contagion
# ---------------------------------------------------------------------------

@dataclass
class ContagionEvent:
    """Records a single contagion event for analysis."""
    source_agent_id: str
    recipient_agent_id: str
    emotion_type: EmotionType
    transferred_intensity: float
    turn: int


class EmotionalContagion:
    """Implements emotional contagion between co-located agents."""

    BASE_PROBABILITY: float = 0.3
    INTENSITY_REDUCTION: float = 0.4   # Contagion intensity = source * this factor

    def run_pass(
        self,
        agents: List[Agent],
        turn: int,
        rng: Optional[random.Random] = None,
    ) -> List[ContagionEvent]:
        """Run one contagion pass over all agents grouped by region.

        Returns a list of ContagionEvent objects for the turn snapshot.
        """
        if rng is None:
            rng = random.Random()

        # Group agents by region
        by_region: Dict[str, List[Agent]] = {}
        for agent in agents:
            by_region.setdefault(agent.region, []).append(agent)

        events: List[ContagionEvent] = []

        for region, region_agents in by_region.items():
            if len(region_agents) < 2:
                continue
            for source in region_agents:
                for recipient in region_agents:
                    if source.agent_id == recipient.agent_id:
                        continue
                    evts = self._attempt_contagion(source, recipient, turn, rng)
                    events.extend(evts)

        return events

    def _attempt_contagion(
        self,
        source: Agent,
        recipient: Agent,
        turn: int,
        rng: random.Random,
    ) -> List[ContagionEvent]:
        events: List[ContagionEvent] = []

        for emotion_type, emotion in source.emotions.items():
            if emotion.intensity < 0.1:
                continue

            contagion_mult = CONTAGION_MULTIPLIERS.get(emotion_type, 0.5)
            extraversion = source.personality.extraversion
            proximity_factor = 1.0  # Single-region proximity assumed

            prob = (
                emotion.intensity
                * extraversion
                * self.BASE_PROBABILITY
                * proximity_factor
                * contagion_mult
            )

            # Susceptibility modifiers for recipient
            susceptibility = 1.0
            susceptibility *= 1.0 + recipient.personality.neuroticism * 0.5
            susceptibility *= 1.0 - recipient.personality.conscientiousness * 0.3
            susceptibility = max(0.0, susceptibility)

            effective_prob = min(1.0, prob * susceptibility)

            if rng.random() < effective_prob:
                transferred = emotion.intensity * self.INTENSITY_REDUCTION
                recipient.add_emotion(emotion_type, transferred)

                # Memory of catching the emotion
                recipient.add_memory(
                    content=(
                        f"I felt {emotion_type.value} spreading from {source.name} — "
                        f"their {emotion_type.value} was palpable."
                    ),
                    importance=0.4,
                    turn=turn,
                )

                events.append(
                    ContagionEvent(
                        source_agent_id=source.agent_id,
                        recipient_agent_id=recipient.agent_id,
                        emotion_type=emotion_type,
                        transferred_intensity=transferred,
                        turn=turn,
                    )
                )

        return events


# ---------------------------------------------------------------------------
# Emotional Breaking Points
# ---------------------------------------------------------------------------

# Breaking point configuration: (emotion_type, min_turns_above_threshold, threshold, name)
BREAKING_POINT_CONFIGS: List[Tuple[EmotionType, int, float, str]] = [
    (EmotionType.FEAR, 5, 0.6, "Paranoia"),
    (EmotionType.ANGER, 4, 0.6, "Rage"),
    (EmotionType.GRIEF, 6, 0.6, "Depression"),
    (EmotionType.LONELINESS, 5, 0.6, "Desperation"),
    (EmotionType.PRIDE, 4, 0.6, "Hubris"),
    (EmotionType.ENVY, 5, 0.6, "Obsession"),
]

CATHARSIS_INTENSITY: float = 0.3  # Emotion resets to this after a breaking point


@dataclass
class BreakingPointEvent:
    """Records a breaking point trigger."""
    agent_id: str
    breaking_point_type: str
    emotion_type: EmotionType
    turn: int


class EmotionalBreakingPoint:
    """Detects and applies emotional breaking points."""

    def check_agents(
        self,
        agents: List[Agent],
        all_agents: List[Agent],
        turn: int,
        rng: Optional[random.Random] = None,
    ) -> List[BreakingPointEvent]:
        """Check all agents for breaking points and apply effects.

        Returns a list of BreakingPointEvent objects.
        """
        if rng is None:
            rng = random.Random()

        events: List[BreakingPointEvent] = []
        for agent in agents:
            # Update consecutive-turns-above-threshold counters for this turn
            for emotion_type, emotion in agent.emotions.items():
                if emotion.intensity >= 0.6:
                    agent.high_intensity_turns[emotion_type] = (
                        agent.high_intensity_turns.get(emotion_type, 0) + 1
                    )
                else:
                    agent.high_intensity_turns[emotion_type] = 0

            for config in BREAKING_POINT_CONFIGS:
                emotion_type, min_turns, threshold, bp_name = config
                emotion = agent.get_emotion(emotion_type)
                turns_above = agent.high_intensity_turns.get(emotion_type, 0)
                if emotion.intensity >= threshold and turns_above >= min_turns:
                    evt = self._trigger(agent, emotion_type, bp_name, all_agents, turn, rng)
                    events.append(evt)
                    # Catharsis — reset emotion and its consecutive-high counter
                    emotion.intensity = CATHARSIS_INTENSITY
                    emotion.turns_active = 0
                    agent.high_intensity_turns[emotion_type] = 0

        return events

    def _trigger(
        self,
        agent: Agent,
        emotion_type: EmotionType,
        bp_name: str,
        all_agents: List[Agent],
        turn: int,
        rng: random.Random,
    ) -> BreakingPointEvent:
        """Apply breaking point effects for the given agent."""
        agent.breaking_point_history.append((bp_name, turn))

        if bp_name == "Paranoia":
            # Trust drops by 0.15 toward ALL agents
            for rel in agent.relationships.values():
                rel.trust = max(0.0, rel.trust - 0.15)
            agent.refuses_trade_turns = max(agent.refuses_trade_turns, 2)
            agent.mood.stability = max(0.0, agent.mood.stability - 0.2)
            agent.add_memory("I can't trust anyone anymore.", importance=0.8, turn=turn)

        elif bp_name == "Rage":
            # Sets attack intent toward the lowest-affinity agent via an aggression boost;
            # the turn engine executes the actual attack during action resolution.
            target_id = agent.lowest_affinity_agent()
            if target_id:
                agent.add_memory("Something snapped inside me.", importance=0.85, turn=turn)
            agent.aggression_boost = 0.3
            agent.aggression_boost_turns = max(agent.aggression_boost_turns, 3)

        elif bp_name == "Depression":
            agent.reduced_action_turns = max(agent.reduced_action_turns, 3)
            agent.silent_turns = max(agent.silent_turns, 3)
            # Reduce needs satisfaction
            agent.needs.safety = max(0.0, agent.needs.safety - 0.2)
            agent.needs.belonging = max(0.0, agent.needs.belonging - 0.2)
            agent.needs.sustenance = max(0.0, agent.needs.sustenance - 0.2)
            agent.add_memory("I can barely find the will to go on.", importance=0.85, turn=turn)

        elif bp_name == "Desperation":
            agent.needs.belonging = 1.0
            agent.add_memory("I'll do anything not to be alone.", importance=0.8, turn=turn)

        elif bp_name == "Hubris":
            agent.needs.safety = max(0.0, agent.needs.safety - 0.2)
            agent.add_memory("I don't need anyone.", importance=0.75, turn=turn)

        elif bp_name == "Obsession":
            # Set obsession target to highest-envy relationship (approximated as lowest affinity)
            target_id = agent.lowest_affinity_agent()
            if target_id:
                agent.obsession_target = target_id
                # 50% chance per turn to steal/sabotage (flag set, engine handles actual action)
            agent.add_memory(
                f"I can't stop thinking about what {agent.obsession_target or 'them'} has.",
                importance=0.8,
                turn=turn,
            )

        # Witnesses form strong memories
        for witness in all_agents:
            if witness.agent_id == agent.agent_id:
                continue
            if witness.region == agent.region:
                witness.add_memory(
                    content=f"I witnessed {agent.name} experience a {bp_name} breaking point — it was unsettling.",
                    importance=0.9,
                    turn=turn,
                )

        return BreakingPointEvent(
            agent_id=agent.agent_id,
            breaking_point_type=bp_name,
            emotion_type=emotion_type,
            turn=turn,
        )


# ---------------------------------------------------------------------------
# Complex Emotion Detector
# ---------------------------------------------------------------------------

# Complex emotion definitions.
# Each entry: (name, [(EmotionType, min_intensity), ...], extra_condition_key, behavioral_effect)
# extra_condition_key is handled in _check_extra_condition().
_COMPLEX_EMOTION_DEFS: List[Tuple[str, List[Tuple[EmotionType, float]], str, str]] = [
    (
        "jealousy",
        [(EmotionType.ENVY, 0.3), (EmotionType.FEAR, 0.2)],
        "",
        "You feel threatened by someone else's relationship with your ally. "
        "You may become possessive or try to sabotage rival relationships.",
    ),
    (
        "contempt",
        [(EmotionType.ANGER, 0.3), (EmotionType.DISGUST, 0.2)],
        "",
        "You look down on this agent with disdain. "
        "You are reluctant to interact and may publicly mock them.",
    ),
    (
        "awe",
        [(EmotionType.FEAR, 0.2), (EmotionType.GRATITUDE, 0.2)],
        "high_respect",
        "You are deeply impressed and slightly intimidated. "
        "You are inclined to follow this agent's suggestions and may join their faction.",
    ),
    (
        "nostalgia",
        [(EmotionType.GRIEF, 0.2), (EmotionType.HOPE, 0.2)],
        "old_memories",
        "You long for better times. "
        "You are motivated to rebuild and may try to reform dissolved factions.",
    ),
    (
        "shame",
        [(EmotionType.FEAR, 0.2), (EmotionType.ANGER, 0.2)],
        "recent_failure",
        "You feel deeply embarrassed about what you did. "
        "You may overcompensate with generosity and will be quieter for a while.",
    ),
    (
        "admiration",
        [(EmotionType.GRATITUDE, 0.3), (EmotionType.HOPE, 0.2)],
        "moderate_respect",
        "You deeply respect and want to emulate this agent. "
        "You learn faster when taught by them and may adopt their beliefs.",
    ),
]


class ComplexEmotionDetector:
    """Detects composite emotions from base emotion combinations."""

    def detect(self, agent: Agent, current_turn: int = 0) -> List[ComplexEmotion]:
        """Return all complex emotions currently active for the agent."""
        detected: List[ComplexEmotion] = []

        for name, components, extra_key, effect in _COMPLEX_EMOTION_DEFS:
            # Check all component emotions meet minimum intensities
            all_met = all(
                agent.get_emotion(et).intensity >= min_intensity
                for et, min_intensity in components
            )
            if not all_met:
                continue

            # Check extra condition
            if extra_key and not self._check_extra_condition(extra_key, agent, current_turn):
                continue

            # Derive intensity from the average of component intensities
            avg_intensity = sum(
                agent.get_emotion(et).intensity for et, _ in components
            ) / len(components)

            detected.append(
                ComplexEmotion(
                    name=name,
                    component_emotions=components,
                    intensity=round(avg_intensity, 3),
                    behavioral_effect=effect,
                )
            )

        return detected

    def _check_extra_condition(
        self, key: str, agent: Agent, current_turn: int
    ) -> bool:
        if key == "high_respect":
            return any(r.respect > 0.7 for r in agent.relationships.values())
        if key == "moderate_respect":
            return any(r.respect > 0.6 for r in agent.relationships.values())
        if key == "old_memories":
            return any(
                (current_turn - m.turn_created) > 10 and m.importance > 0.5
                for m in agent.memories
            )
        if key == "recent_failure":
            # Approximate: check if there are any memories from the last 3 turns
            # (a recent event that could represent a norm violation or failed action)
            recent = [
                m for m in agent.memories
                if (current_turn - m.turn_created) <= 3
            ]
            return len(recent) > 0
        return True
