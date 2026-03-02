from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from models.world_state import WorldState


def compute_gini(values: List[float]) -> float:
    """Standard Gini coefficient calculation."""
    if not values:
        return 0.0
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumsum = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def compute_alliance_density(world_state: "WorldState") -> float:
    """Fraction of agent pairs with trust > 0.6."""
    alive = [a for a in world_state.agents.values() if a.is_alive]
    n = len(alive)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    alliance_pairs = 0
    for i, a1 in enumerate(alive):
        for a2 in alive[i + 1:]:
            rel = a1.relationships.get(a2.id)
            if rel and rel.trust > 0.6:
                alliance_pairs += 1
    return alliance_pairs / total_pairs


def compute_innovation_rate(turn_history: List[Dict]) -> float:
    """Count turns with at least one new concept discovered / total turns."""
    if not turn_history:
        return 0.0
    turns_with_discovery = sum(
        1 for t in turn_history if t.get("new_concepts_discovered", 0) > 0
    )
    return turns_with_discovery / len(turn_history)


def compute_conflict_frequency(turn_history: List[Dict]) -> float:
    """Average number of combat actions per turn."""
    if not turn_history:
        return 0.0
    total_combat = sum(t.get("combat_actions", 0) for t in turn_history)
    return total_combat / len(turn_history)


def compute_survival_rate(world_state: "WorldState") -> float:
    """Alive agents / total agents."""
    total = len(world_state.agents)
    if total == 0:
        return 0.0
    alive = sum(1 for a in world_state.agents.values() if a.is_alive)
    return alive / total


def compute_metrics(world_state: "WorldState", turn_history: List[Dict]) -> Dict:
    alive = [a for a in world_state.agents.values() if a.is_alive]
    food_values = [a.inventory.food for a in alive]
    return {
        "gini": compute_gini(food_values),
        "alliance_density": compute_alliance_density(world_state),
        "innovation_rate": compute_innovation_rate(turn_history),
        "conflict_frequency": compute_conflict_frequency(turn_history),
        "survival_rate": compute_survival_rate(world_state),
        "alive_count": len(alive),
        "faction_count": len(world_state.factions),
        "structure_count": sum(len(r.structures) for r in world_state.regions.values()),
    }
