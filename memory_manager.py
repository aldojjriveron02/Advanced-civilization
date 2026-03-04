"""
memory_manager.py — Manages agent memory retrieval and context building.

Supports emotional weighting: memories formed during high-emotion states are
recalled more vividly and resist pruning.
"""

from __future__ import annotations

from typing import List, Optional

from agent_state import MemoryEntry


class MemoryManager:
    """Handles storage, retrieval, and context building for agent memories."""

    MAX_MEMORIES: int = 200          # Hard cap on memory list length
    PRUNE_TARGET: int = 150          # Prune down to this many memories
    VIVID_TOP_N: int = 5             # Number of vivid memories shown in context
    EMOTIONAL_RECALL_BONUS: float = 0.25  # Multiplier on emotional_intensity bonus

    def __init__(self, memories: List[MemoryEntry]) -> None:
        """
        Args:
            memories: Reference to the agent's memory list (mutated in-place).
        """
        self._memories = memories

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_importance_threshold: float = 0.0,
        current_turn: int = 0,
        top_n: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Return memories sorted by a composite recall score.

        Scoring:
          base_score = memory.importance
          recency_bonus = 1 / (1 + age_in_turns * 0.1)
          emotional_recall_bonus = memory.emotional_intensity_at_creation * EMOTIONAL_RECALL_BONUS
          final_score = base_score + recency_bonus + emotional_recall_bonus
        """
        scored: List[tuple[float, MemoryEntry]] = []
        for mem in self._memories:
            if mem.importance < query_importance_threshold:
                continue
            age = max(0, current_turn - mem.turn_created)
            recency = 1.0 / (1.0 + age * 0.1)
            emotional_bonus = mem.emotional_intensity_at_creation * self.EMOTIONAL_RECALL_BONUS
            score = mem.importance + recency + emotional_bonus
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored]
        if top_n is not None:
            results = results[:top_n]
        return results

    def vivid_memories(self) -> List[MemoryEntry]:
        """Return the top N memories that qualify as vivid (emotional_intensity > 0.7),
        ordered by emotional intensity descending."""
        candidates = [m for m in self._memories if m.is_vivid()]
        candidates.sort(key=lambda m: m.emotional_intensity_at_creation, reverse=True)
        return candidates[: self.VIVID_TOP_N]

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context_summary(self, current_turn: int = 0, top_n: int = 10) -> str:
        """Build a human-readable memory context string for LLM prompts."""
        recent = self.retrieve(current_turn=current_turn, top_n=top_n)
        vivid = self.vivid_memories()

        lines: List[str] = []

        if recent:
            lines.append("=== RECENT MEMORIES ===")
            for mem in recent:
                prefix = "[VIVID] " if mem.is_vivid() else ""
                lines.append(f"  {prefix}[importance={mem.importance:.2f}] {mem.content}")

        if vivid:
            lines.append("=== EMOTIONALLY VIVID MEMORIES ===")
            for mem in vivid:
                lines.append(
                    f"  [VIVID] [intensity={mem.emotional_intensity_at_creation:.2f}] {mem.content}"
                )

        return "\n".join(lines) if lines else "(No memories yet)"

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_if_needed(self, current_turn: int = 0) -> int:
        """Remove low-importance, low-emotion memories if over MAX_MEMORIES.

        Returns the number of memories removed.
        """
        if len(self._memories) <= self.MAX_MEMORIES:
            return 0

        # Score each memory for survival; vivid memories are harder to prune
        def prune_score(mem: MemoryEntry) -> float:
            age = max(0, current_turn - mem.turn_created)
            recency = 1.0 / (1.0 + age * 0.1)
            return mem.importance + recency + mem.emotional_intensity_at_creation * 0.5

        scored = sorted(self._memories, key=prune_score, reverse=True)
        removed = len(self._memories) - self.PRUNE_TARGET
        self._memories[:] = scored[: self.PRUNE_TARGET]
        return removed
