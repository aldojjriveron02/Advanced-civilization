"""
turn_engine.py — Orchestrates all simulation phases for each turn.

Updated turn execution order (phases 1-18):
 1-3.  Environment, events, needs tick (stubs)
 4.    Agent observation
 5.    Agent reasoning + action declaration (with mood + complex emotions in prompt)
 6-8.  Action conversion, conflict detection, resolution (stubs)
 9.    Norm compliance check (stub)
10.    Communication phase (with EmotionalSpeechModifier)
11.    Emotional contagion pass  [NEW]
12.    Faction update (stub)
13.    Survival check (stub)
14.    Breaking point check      [NEW]
15.    Complex emotion detection [NEW]
16.    Mood update               [NEW]
17.    Relationship decay
18.    Metrics computation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_state import Agent, Emotion
from communication_parser import CommunicationPipeline, Message, CommunicationResult
from emotion_system import (
    EmotionalContagion,
    ContagionEvent,
    EmotionalBreakingPoint,
    BreakingPointEvent,
    ComplexEmotionDetector,
)
from memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Turn snapshot
# ---------------------------------------------------------------------------

@dataclass
class TurnSnapshot:
    """Records key events and metrics from a single turn."""
    turn: int
    contagion_events: List[ContagionEvent] = field(default_factory=list)
    breaking_point_events: List[BreakingPointEvent] = field(default_factory=list)
    communication_results: List[CommunicationResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Turn engine
# ---------------------------------------------------------------------------

class TurnEngine:
    """Runs all simulation phases for one turn."""

    def __init__(self, agents: List[Agent], seed: Optional[int] = None) -> None:
        self.agents = agents
        self.agents_by_id: Dict[str, Agent] = {a.agent_id: a for a in agents}
        self.rng = random.Random(seed)
        self.turn = 0

        # Sub-systems
        self._contagion = EmotionalContagion()
        self._breaking_point = EmotionalBreakingPoint()
        self._complex_detector = ComplexEmotionDetector()
        self._comm_pipeline = CommunicationPipeline()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_turn(self, messages: Optional[List[Message]] = None) -> TurnSnapshot:
        """Execute all phases of a single simulation turn."""
        self.turn += 1
        snapshot = TurnSnapshot(turn=self.turn)

        # Phase 1-3: Environment, events, needs tick (stubs)
        self._phase_environment()
        self._phase_events()
        self._phase_needs_tick()

        # Phase 4: Agent observation
        self._phase_observation()

        # Phase 5: Agent reasoning + action declaration
        self._phase_reasoning()

        # Phase 6-8: Action conversion, conflict detection, resolution (stubs)
        self._phase_actions()

        # Phase 9: Norm compliance check (stub)
        self._phase_norm_compliance()

        # Phase 10: Communication phase
        comm_results = self._phase_communication(messages or [])
        snapshot.communication_results = comm_results

        # Phase 11: Emotional contagion pass [NEW]
        contagion_events = self._phase_emotional_contagion()
        snapshot.contagion_events = contagion_events

        # Phase 12: Faction update (stub)
        self._phase_faction_update()

        # Phase 13: Survival check (stub)
        self._phase_survival_check()

        # Phase 14: Breaking point check [NEW]
        breaking_events = self._phase_breaking_points()
        snapshot.breaking_point_events = breaking_events

        # Phase 15: Complex emotion detection [NEW]
        self._phase_complex_emotions()

        # Phase 16: Mood update [NEW]
        self._phase_mood_update()

        # Phase 17: Relationship decay
        self._phase_relationship_decay()

        # Phase 18: Metrics computation
        snapshot.metrics = self._compute_metrics(contagion_events, breaking_events)

        # Decay emotions at end of turn
        for agent in self.agents:
            agent.tick_emotions()
            # Prune memories if needed
            mm = MemoryManager(agent.memories)
            mm.prune_if_needed(current_turn=self.turn)

        return snapshot

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_environment(self) -> None:
        """Phase 1: Update environmental state (stub)."""
        pass

    def _phase_events(self) -> None:
        """Phase 2: Trigger world events (stub)."""
        pass

    def _phase_needs_tick(self) -> None:
        """Phase 3: Tick agent needs (stub — subclass to implement)."""
        pass

    def _phase_observation(self) -> None:
        """Phase 4: Each agent observes their surroundings (stub)."""
        pass

    def _phase_reasoning(self) -> None:
        """Phase 5: Agent reasoning with mood + complex emotions (stub)."""
        pass

    def _phase_actions(self) -> None:
        """Phases 6-8: Action processing (stub)."""
        pass

    def _phase_norm_compliance(self) -> None:
        """Phase 9: Norm compliance check (stub)."""
        pass

    def _phase_communication(
        self, messages: List[Message]
    ) -> List[CommunicationResult]:
        """Phase 10: Process all agent communications."""
        for msg in messages:
            msg.turn = self.turn
        return self._comm_pipeline.process_turn_messages(messages, self.agents_by_id)

    def _phase_emotional_contagion(self) -> List[ContagionEvent]:
        """Phase 11: Emotional contagion pass."""
        return self._contagion.run_pass(self.agents, turn=self.turn, rng=self.rng)

    def _phase_faction_update(self) -> None:
        """Phase 12: Faction updates (stub)."""
        pass

    def _phase_survival_check(self) -> None:
        """Phase 13: Survival checks (stub)."""
        pass

    def _phase_breaking_points(self) -> List[BreakingPointEvent]:
        """Phase 14: Emotional breaking point checks."""
        return self._breaking_point.check_agents(
            self.agents, self.agents, self.turn, rng=self.rng
        )

    def _phase_complex_emotions(self) -> None:
        """Phase 15: Detect complex emotions for all agents."""
        for agent in self.agents:
            agent.complex_emotions = self._complex_detector.detect(
                agent, current_turn=self.turn
            )

    def _phase_mood_update(self) -> None:
        """Phase 16: Update each agent's mood based on recent emotion history (last 5 turns)."""
        for agent in self.agents:
            # Flatten the rolling emotion history into a list of Emotion-like objects.
            # Fall back to the current emotion state if no history exists yet.
            if agent.emotion_history:
                history_emotions = [
                    Emotion(emotion_type=et, intensity=intensity)
                    for snapshot in agent.emotion_history
                    for et, intensity in snapshot.items()
                ]
            else:
                history_emotions = list(agent.emotions.values())
            agent.mood.update(
                recent_emotions=history_emotions,
                needs_satisfied=agent.needs.is_satisfied(),
                social_cooperation=False,   # stub — set by action resolution phase
                social_betrayal=False,       # stub
                health=agent.needs.health,
            )

    def _phase_relationship_decay(self) -> None:
        """Phase 17: Slowly decay relationship trust over time."""
        DECAY = 0.005
        for agent in self.agents:
            for rel in agent.relationships.values():
                rel.trust = max(0.0, rel.trust - DECAY)

    def _compute_metrics(
        self,
        contagion_events: List[ContagionEvent],
        breaking_events: List[BreakingPointEvent],
    ) -> Dict[str, Any]:
        """Phase 18: Compute emotional metrics."""
        avg_optimism = (
            sum(a.mood.optimism for a in self.agents) / len(self.agents)
            if self.agents else 0.0
        )
        return {
            "turn": self.turn,
            "avg_mood_optimism": round(avg_optimism, 4),
            "contagion_event_count": len(contagion_events),
            "breaking_point_count": len(breaking_events),
            "breaking_point_types": [e.breaking_point_type for e in breaking_events],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents_by_id.get(agent_id)
