from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING


@dataclass
class FactionNorm:
    description: str
    compliance_count: int = 0
    violation_count: int = 0

    @property
    def strength(self) -> float:
        return self.compliance_count / (self.compliance_count + self.violation_count + 1)


@dataclass
class Faction:
    id: str
    name: str
    founder_id: str
    members: List[str] = field(default_factory=list)
    leader_id: str = ""
    norms: List[FactionNorm] = field(default_factory=list)
    shared_beliefs: List[str] = field(default_factory=list)
    territory_claims: List[str] = field(default_factory=list)
    formed_turn: int = 0
    internal_cohesion: float = 0.5
    external_reputation: float = 0.5

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_viable(self) -> bool:
        return self.size >= 2 and self.internal_cohesion >= 0.25

    def add_member(self, agent_id: str) -> None:
        if agent_id not in self.members:
            self.members.append(agent_id)

    def remove_member(self, agent_id: str) -> None:
        if agent_id in self.members:
            self.members.remove(agent_id)

    def update_cohesion(self, world_state: Any) -> None:
        """Compute average trust between member pairs from agent relationships."""
        if len(self.members) < 2:
            self.internal_cohesion = 0.0
            return
        trust_values = []
        for i, m1 in enumerate(self.members):
            for m2 in self.members[i + 1:]:
                agent1 = world_state.agents.get(m1)
                agent2 = world_state.agents.get(m2)
                if agent1 and agent2:
                    rel = agent1.relationships.get(m2)
                    if rel:
                        trust_values.append(rel.trust)
                    rel2 = agent2.relationships.get(m1)
                    if rel2:
                        trust_values.append(rel2.trust)
        self.internal_cohesion = sum(trust_values) / len(trust_values) if trust_values else 0.5
