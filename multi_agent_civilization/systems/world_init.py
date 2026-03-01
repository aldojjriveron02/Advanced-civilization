from __future__ import annotations
import random as _random
from typing import Optional

from models.world_state import (
    WorldState, Region, RegionType, ResourceType, ResourceNode, Season
)
from models.agent_state import (
    Agent, Archetype, PersonalityTraits, Inventory, Needs, ARCHETYPE_PRESETS
)


def create_default_world(seed: Optional[int] = None) -> WorldState:
    rng = _random.Random(seed)
    world = WorldState()

    # --- Regions ---
    world.regions["forest"] = Region(
        id="forest", name="Verdant Forest", region_type=RegionType.FOREST,
        resources={
            ResourceType.FOOD: ResourceNode(ResourceType.FOOD, 15.0, 20.0, 0.5),
            ResourceType.WOOD: ResourceNode(ResourceType.WOOD, 20.0, 30.0, 0.3),
            ResourceType.HERBS: ResourceNode(ResourceType.HERBS, 10.0, 15.0, 0.4),
        },
        defensibility=0.4, flood_risk=0.05, fire_risk=0.2, is_discovered=True,
    )
    world.regions["river"] = Region(
        id="river", name="Silver River", region_type=RegionType.RIVER,
        resources={
            ResourceType.FOOD: ResourceNode(ResourceType.FOOD, 12.0, 18.0, 0.6),
            ResourceType.WATER: ResourceNode(ResourceType.WATER, 25.0, 30.0, 0.8),
        },
        defensibility=0.3, flood_risk=0.3, fire_risk=0.02, is_discovered=True,
    )
    world.regions["hills"] = Region(
        id="hills", name="Iron Hills", region_type=RegionType.HILLS,
        resources={
            ResourceType.STONE: ResourceNode(ResourceType.STONE, 20.0, 25.0, 0.2),
            ResourceType.METAL: ResourceNode(ResourceType.METAL, 10.0, 15.0, 0.1),
        },
        defensibility=0.7, flood_risk=0.02, fire_risk=0.05, is_discovered=True,
    )
    world.regions["coast"] = Region(
        id="coast", name="Hidden Coast", region_type=RegionType.COAST,
        resources={
            ResourceType.FOOD: ResourceNode(ResourceType.FOOD, 10.0, 15.0, 0.5),
            ResourceType.WATER: ResourceNode(ResourceType.WATER, 20.0, 25.0, 0.7),
        },
        defensibility=0.35, flood_risk=0.2, fire_risk=0.02, is_discovered=False,
    )
    world.regions["caves"] = Region(
        id="caves", name="Volcanic Caves", region_type=RegionType.VOLCANIC,
        resources={
            ResourceType.METAL: ResourceNode(ResourceType.METAL, 15.0, 20.0, 0.15),
            ResourceType.STONE: ResourceNode(ResourceType.STONE, 18.0, 22.0, 0.15),
        },
        defensibility=0.8, flood_risk=0.01, fire_risk=0.1, is_discovered=False,
    )

    discovered_regions = [rid for rid, r in world.regions.items() if r.is_discovered]

    # --- Agent definitions: (name, archetype, starting_region) ---
    agent_defs = [
        ("Kael", Archetype.BUILDER, "forest"),
        ("Vex", Archetype.OPPORTUNIST, "forest"),
        ("Thorn", Archetype.PROTECTOR, "forest"),
        ("Miriel", Archetype.DIPLOMAT, "river"),
        ("Senna", Archetype.HEALER, "river"),
        ("Lyra", Archetype.INVENTOR, "river"),
        ("Asha", Archetype.ALTRUIST, "river"),
        ("Rhett", Archetype.EXPLORER, "hills"),
        ("Grunn", Archetype.HOARDER, "hills"),
        ("Draven", Archetype.STRATEGIST, "hills"),
    ]

    archetype_skill_bonuses = {
        Archetype.BUILDER: {"building": 0.3, "crafting": 0.2},
        Archetype.DIPLOMAT: {"persuasion": 0.4},
        Archetype.OPPORTUNIST: {"gathering": 0.2, "combat": 0.2},
        Archetype.HEALER: {"medicine": 0.4, "gathering": 0.1},
        Archetype.EXPLORER: {"exploration": 0.4, "gathering": 0.2},
        Archetype.HOARDER: {"gathering": 0.3},
        Archetype.INVENTOR: {"crafting": 0.3, "building": 0.1},
        Archetype.PROTECTOR: {"combat": 0.4, "building": 0.1},
        Archetype.ALTRUIST: {"medicine": 0.1, "farming": 0.2},
        Archetype.STRATEGIST: {"persuasion": 0.2, "combat": 0.2},
    }

    for name, archetype, start_region in agent_defs:
        agent_id = name.lower()
        preset = ARCHETYPE_PRESETS[archetype]

        # Build personality with preset values + ±0.1 variance
        trait_fields = PersonalityTraits.__dataclass_fields__.keys()
        trait_values = {}
        for trait in trait_fields:
            base = preset.get(trait, 0.5)
            variance = rng.uniform(-0.1, 0.1)
            trait_values[trait] = max(0.0, min(1.0, base + variance))

        personality = PersonalityTraits(**trait_values)

        # Build skills
        skill_names = ["gathering", "building", "combat", "medicine", "crafting",
                       "persuasion", "exploration", "farming"]
        skills = {s: 0.3 + rng.uniform(0, 0.2) for s in skill_names}
        for skill, bonus in archetype_skill_bonuses.get(archetype, {}).items():
            skills[skill] = min(1.0, skills[skill] + bonus)

        agent = Agent(
            id=agent_id,
            name=name,
            archetype=archetype,
            personality=personality,
            skills=skills,
            current_region=start_region,
            known_regions=list({start_region} | set(discovered_regions)),
        )
        agent.inventory.food = rng.uniform(3, 6)
        agent.inventory.water = rng.uniform(3, 6)

        world.agents[agent_id] = agent
        world.regions[start_region].occupants.append(agent_id)

    return world
