from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models.world_state import Season, WeatherType, RegionType, ResourceType

if TYPE_CHECKING:
    from models.world_state import WorldState


@dataclass
class WorldEvent:
    name: str
    description: str
    probability: float
    cooldown: int
    season_restriction: Optional[List[Season]] = None
    min_turn: int = 0
    last_triggered: int = -999


class EventSystem:
    def __init__(self) -> None:
        self.events: List[WorldEvent] = [
            WorldEvent("Flood", "Rising waters flood low-lying areas.", 0.06, 10,
                       season_restriction=[Season.SPRING, Season.AUTUMN]),
            WorldEvent("Wildfire", "A wildfire sweeps through forest regions.", 0.04, 15,
                       season_restriction=[Season.SUMMER]),
            WorldEvent("Bountiful Harvest", "Nature blesses the land with abundance.", 0.08, 8,
                       season_restriction=[Season.AUTUMN]),
            WorldEvent("Harsh Blizzard", "A brutal blizzard covers the land.", 0.10, 6,
                       season_restriction=[Season.WINTER]),
            WorldEvent("Disease Outbreak", "A sickness spreads among the population.", 0.03, 20,
                       min_turn=15),
            WorldEvent("Resource Discovery", "A new cache of resources is found.", 0.05, 12,
                       min_turn=10),
            WorldEvent("New Region Revealed", "Explorers reveal an unknown region.", 0.03, 20,
                       min_turn=20),
            WorldEvent("Animal Migration", "Herds of animals pass through.", 0.07, 10,
                       season_restriction=[Season.SPRING, Season.AUTUMN]),
            WorldEvent("Earthquake", "The ground shakes violently.", 0.02, 25, min_turn=15),
            WorldEvent("Solar Eclipse", "The sun darkens and fear grips hearts.", 0.01, 50,
                       min_turn=30),
        ]

    def roll_events(self, world_state: "WorldState", rng: Any) -> List[Dict]:
        triggered = []
        for event in self.events:
            age = world_state.turn - event.last_triggered
            if age < event.cooldown:
                continue
            if world_state.turn < event.min_turn:
                continue
            if event.season_restriction and world_state.season not in event.season_restriction:
                continue
            if rng.random() < event.probability:
                event.last_triggered = world_state.turn
                effects = self._apply_event(event.name, world_state, rng)
                triggered.append({"name": event.name, "description": event.description, "effects": effects})
                world_state.event_log.append(f"Turn {world_state.turn}: {event.name} - {event.description}")
        return triggered

    def _apply_event(self, name: str, world_state: "WorldState", rng: Any) -> Dict:
        dispatch = {
            "Flood": self._apply_flood,
            "Wildfire": self._apply_wildfire,
            "Bountiful Harvest": self._apply_bountiful_harvest,
            "Harsh Blizzard": self._apply_blizzard,
            "Disease Outbreak": self._apply_disease,
            "Resource Discovery": lambda ws: self._apply_resource_discovery(ws, rng),
            "New Region Revealed": self._apply_new_region,
            "Animal Migration": self._apply_migration,
            "Earthquake": self._apply_earthquake,
            "Solar Eclipse": self._apply_eclipse,
        }
        fn = dispatch.get(name)
        if fn:
            return fn(world_state)
        return {}

    def _apply_flood(self, world_state: "WorldState") -> Dict:
        affected = []
        for region in world_state.regions.values():
            if region.region_type == RegionType.RIVER:
                node = region.resources.get(ResourceType.FOOD)
                if node:
                    node.amount *= 0.7
                for struct in region.structures:
                    struct.durability = max(0.0, struct.durability - 0.2)
                affected.append(region.name)
        return {"type": "flood", "affected_regions": affected}

    def _apply_wildfire(self, world_state: "WorldState") -> Dict:
        affected = []
        for region in world_state.regions.values():
            if region.region_type == RegionType.FOREST:
                node = region.resources.get(ResourceType.WOOD)
                if node:
                    node.amount *= 0.5
                region.environmental_scars.append("wildfire_char")
                affected.append(region.name)
        return {"type": "wildfire", "affected_regions": affected}

    def _apply_bountiful_harvest(self, world_state: "WorldState") -> Dict:
        for region in world_state.regions.values():
            node = region.resources.get(ResourceType.FOOD)
            if node:
                node.amount = min(node.max_amount, node.amount * 1.5)
        return {"type": "bountiful_harvest"}

    def _apply_blizzard(self, world_state: "WorldState") -> Dict:
        damaged = []
        for agent in world_state.agents.values():
            if not agent.is_alive:
                continue
            region = world_state.regions.get(agent.current_region)
            has_shelter = region and any(
                s.is_complete and s.name in ("shelter", "wall") for s in region.structures
            )
            if not has_shelter:
                agent.health = max(0.0, agent.health - 0.15)
                damaged.append(agent.id)
        return {"type": "blizzard", "damaged_agents": damaged}

    def _apply_disease(self, world_state: "WorldState") -> Dict:
        import random as _random
        alive = [a for a in world_state.agents.values() if a.is_alive]
        damaged = []
        for agent in alive:
            dmg = 0.1 + _random.random() * 0.1
            agent.health = max(0.0, agent.health - dmg)
            damaged.append(agent.id)
        return {"type": "disease", "damaged_agents": damaged}

    def _apply_resource_discovery(self, world_state: "WorldState", rng: Any) -> Dict:
        discovered_regions = [r for r in world_state.regions.values() if r.is_discovered]
        if not discovered_regions:
            return {}
        region = rng.choice(discovered_regions)
        resource_types = [ResourceType.METAL, ResourceType.STONE, ResourceType.HERBS]
        rt = rng.choice(resource_types)
        if rt in region.resources:
            region.resources[rt].amount = min(
                region.resources[rt].max_amount,
                region.resources[rt].amount + 5.0
            )
        return {"type": "resource_discovery", "region": region.id, "resource": rt.value}

    def _apply_new_region(self, world_state: "WorldState") -> Dict:
        hidden = [r for r in world_state.regions.values() if not r.is_discovered]
        if hidden:
            region = hidden[0]
            region.is_discovered = True
            for agent in world_state.agents.values():
                if agent.is_alive and region.id not in agent.known_regions:
                    agent.known_regions.append(region.id)
            return {"type": "region_revealed", "region": region.id}
        return {"type": "region_revealed", "region": None}

    def _apply_migration(self, world_state: "WorldState") -> Dict:
        for region in world_state.regions.values():
            node = region.resources.get(ResourceType.FOOD)
            if node:
                node.amount = min(node.max_amount, node.amount + 3.0)
        return {"type": "migration"}

    def _apply_earthquake(self, world_state: "WorldState") -> Dict:
        for region in world_state.regions.values():
            if region.region_type == RegionType.HILLS:
                for struct in region.structures:
                    struct.durability = max(0.0, struct.durability - 0.3)
                region.environmental_scars.append("earthquake_crack")
        return {"type": "earthquake"}

    def _apply_eclipse(self, world_state: "WorldState") -> Dict:
        from models.agent_state import Belief
        import random as _random
        alive = [a for a in world_state.agents.values() if a.is_alive]
        for agent in _random.sample(alive, min(3, len(alive))):
            agent.beliefs.append(Belief(
                content="The darkened sun is an omen of change",
                confidence=0.6,
                source="eclipse_event",
                falsifiability=0.1,
            ))
        return {"type": "eclipse"}
