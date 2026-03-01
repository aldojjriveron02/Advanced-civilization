from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from models.agent_state import Agent, MemoryEntry


@dataclass
class MemoryQuery:
    keywords: List[str] = field(default_factory=list)
    related_agents: List[str] = field(default_factory=list)
    category: Optional[str] = None
    min_importance: float = 0.0
    max_age: Optional[int] = None
    emotional_valence_range: Optional[Tuple[float, float]] = None
    limit: int = 10


class MemoryManager:
    def retrieve(self, agent: "Agent", query: MemoryQuery, current_turn: int) -> List["MemoryEntry"]:
        results = []
        for entry in agent.memories:
            age = current_turn - entry.turn
            if query.category and entry.category != query.category:
                continue
            if entry.importance < query.min_importance:
                continue
            if query.max_age is not None and age > query.max_age:
                continue
            if query.emotional_valence_range is not None:
                lo, hi = query.emotional_valence_range
                if not (lo <= entry.emotional_valence <= hi):
                    continue

            score = 0.0
            # agent relevance
            overlap = len(set(entry.related_agents) & set(query.related_agents))
            score += 0.3 * overlap
            # keyword match
            content_lower = entry.content.lower()
            for kw in query.keywords:
                if kw.lower() in content_lower:
                    score += 0.2
            # recency
            score += math.exp(-age / 20)
            # importance
            score += 0.3 * entry.importance
            # emotional intensity
            score += 0.1 * abs(entry.emotional_valence)

            results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[: query.limit]]

    def build_context_summary(self, agent: "Agent", current_turn: int) -> str:
        lines = []

        # Recent memories
        lines.append("RECENT MEMORIES (last 5 turns):")
        recent = [m for m in agent.memories if current_turn - m.turn <= 5]
        recent_sorted = sorted(recent, key=lambda m: m.turn, reverse=True)[:15]
        for m in recent_sorted:
            lines.append(f"  [T{m.turn}] {m.content}")

        # Key long-term memories
        lines.append("KEY LONG-TERM MEMORIES:")
        important = [m for m in agent.memories if m.importance >= 0.7]
        important_sorted = sorted(important, key=lambda m: m.importance, reverse=True)[:10]
        for m in important_sorted:
            lines.append(f"  [{m.category}] {m.content}")

        # Relationship summary
        lines.append("RELATIONSHIP SUMMARY:")
        for agent_id, rel in agent.relationships.items():
            lines.append(
                f"  {agent_id}: trust={rel.trust:.2f}, affinity={rel.affinity:.2f}, "
                f"fear={rel.fear:.2f}, debt={rel.debt:.1f}"
            )

        # Active beliefs
        lines.append("ACTIVE BELIEFS:")
        sorted_beliefs = sorted(agent.beliefs, key=lambda b: b.confidence, reverse=True)[:8]
        for belief in sorted_beliefs:
            lines.append(f"  - {belief.content} (confidence: {belief.confidence:.2f})")

        return "\n".join(lines)
