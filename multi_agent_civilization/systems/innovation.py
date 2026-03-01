from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from models.agent_state import Agent
    from models.world_state import WorldState


@dataclass
class KnowledgeConcept:
    name: str
    tier: int
    prerequisites: List[str] = field(default_factory=list)
    discovery_source: str = "action"


@dataclass
class Invention:
    name: str
    required_concepts: List[str]
    required_resources: Dict[str, float] = field(default_factory=dict)
    min_collaborators: int = 0
    effect_type: str = ""
    effects: Dict = field(default_factory=dict)


class InnovationSystem:
    def __init__(self) -> None:
        self.CONCEPTS: Dict[str, KnowledgeConcept] = {
            # Tier 0
            "fire_making": KnowledgeConcept("fire_making", 0),
            "basic_farming": KnowledgeConcept("basic_farming", 0),
            "stone_working": KnowledgeConcept("stone_working", 0),
            "herbalism": KnowledgeConcept("herbalism", 0),
            "animal_tracking": KnowledgeConcept("animal_tracking", 0),
            "rope_making": KnowledgeConcept("rope_making", 0),
            # Tier 1
            "irrigation": KnowledgeConcept("irrigation", 1, prerequisites=["basic_farming"]),
            "masonry": KnowledgeConcept("masonry", 1, prerequisites=["stone_working"]),
            "cooking": KnowledgeConcept("cooking", 1, prerequisites=["fire_making"]),
            "trapping": KnowledgeConcept("trapping", 1, prerequisites=["animal_tracking"]),
            "basic_medicine": KnowledgeConcept("basic_medicine", 1, prerequisites=["herbalism"]),
            "tool_making": KnowledgeConcept("tool_making", 1, prerequisites=["stone_working"]),
            # Tier 2
            "architecture": KnowledgeConcept("architecture", 2, prerequisites=["masonry"]),
            "metallurgy": KnowledgeConcept("metallurgy", 2, prerequisites=["tool_making", "stone_working"]),
            "advanced_farming": KnowledgeConcept("advanced_farming", 2, prerequisites=["irrigation", "basic_farming"]),
            "surgery": KnowledgeConcept("surgery", 2, prerequisites=["basic_medicine"]),
            # Tier 3
            "writing": KnowledgeConcept("writing", 3, prerequisites=["basic_medicine", "tool_making"]),
            "trade_systems": KnowledgeConcept("trade_systems", 3, prerequisites=["tool_making"]),
            "fortification": KnowledgeConcept("fortification", 3, prerequisites=["architecture", "masonry"]),
            "governance": KnowledgeConcept("governance", 3, prerequisites=["writing", "trade_systems"]),
        }

        self.INVENTIONS: List[Invention] = [
            Invention(
                name="stone_axe",
                required_concepts=["stone_working", "tool_making"],
                required_resources={"stone": 2.0, "wood": 1.0},
                effects={"gathering_bonus": 0.3},
                effect_type="tool",
            ),
            Invention(
                name="granary",
                required_concepts=["basic_farming", "masonry"],
                required_resources={"wood": 6.0, "stone": 3.0},
                effects={"food_storage_bonus": 0.5},
                effect_type="structure",
            ),
            Invention(
                name="aqueduct",
                required_concepts=["irrigation", "masonry"],
                required_resources={"stone": 8.0, "wood": 2.0},
                effects={"water_regen": 0.3},
                effect_type="structure",
            ),
            Invention(
                name="forge",
                required_concepts=["metallurgy"],
                required_resources={"stone": 5.0, "wood": 3.0},
                effects={"metal_efficiency": 0.4},
                effect_type="structure",
            ),
            Invention(
                name="currency",
                required_concepts=["trade_systems"],
                required_resources={},
                effects={"trade_value_bonus": 0.25},
                effect_type="social",
            ),
            Invention(
                name="law_code",
                required_concepts=["governance", "writing"],
                required_resources={},
                effects={"faction_cohesion_bonus": 0.2},
                effect_type="social",
            ),
        ]

        # Action-to-concept mapping for tier 0 discoveries
        self._action_concept_map: Dict[str, List[str]] = {
            "GATHER": ["herbalism", "animal_tracking", "basic_farming"],
            "BUILD": ["stone_working", "rope_making"],
            "CRAFT": ["stone_working", "rope_making", "fire_making"],
            "HEAL": ["herbalism"],
            "EXPLORE": ["animal_tracking"],
            "FARM": ["basic_farming"],
            "REST": ["fire_making"],
        }

    def get_available_inventions(self, agent_knowledge: Set[str]) -> List[Invention]:
        return [
            inv for inv in self.INVENTIONS
            if all(c in agent_knowledge for c in inv.required_concepts)
        ]

    def check_concept_discovery(
        self, agent: "Agent", action_type: str, rng: Any
    ) -> Optional[str]:
        from models.agent_state import Belief
        relevant = self._action_concept_map.get(action_type, [])
        already_known: Set[str] = {
            b.content.replace("Discovered: ", "")
            for b in agent.beliefs
            if b.content.startswith("Discovered: ")
        }

        for concept_name in relevant:
            concept = self.CONCEPTS.get(concept_name)
            if concept is None or concept.tier != 0:
                continue
            if concept_name in already_known:
                continue
            skill_key = {
                "herbalism": "medicine",
                "animal_tracking": "exploration",
                "basic_farming": "farming",
                "stone_working": "crafting",
                "rope_making": "crafting",
                "fire_making": "crafting",
            }.get(concept_name, "gathering")
            prob = agent.skills.get(skill_key, 0.5) * 0.15
            if rng.random() < prob:
                agent.beliefs.append(Belief(
                    content=f"Discovered: {concept_name}",
                    confidence=0.9,
                    source=f"action:{action_type}",
                    falsifiability=0.8,
                ))
                return concept_name
        return None
