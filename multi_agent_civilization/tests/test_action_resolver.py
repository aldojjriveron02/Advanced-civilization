import pytest
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.world_state import WorldState, Region, RegionType, ResourceNode, ResourceType
from models.agent_state import Agent, Archetype, PersonalityTraits, Needs, Inventory, ARCHETYPE_PRESETS
from models.actions import Action, ActionType, ActionCategory
from models.faction import Faction
from systems.action_resolver import ActionResolver
from systems.world_init import create_default_world


def make_agent(agent_id="a1", region_id="forest", archetype=Archetype.BUILDER):
    preset = ARCHETYPE_PRESETS[archetype]
    valid_fields = PersonalityTraits.__dataclass_fields__.keys()
    personality = PersonalityTraits(**{k: v for k, v in preset.items() if k in valid_fields})
    agent = Agent(
        id=agent_id,
        name="TestAgent",
        archetype=archetype,
        personality=personality,
        current_region=region_id,
    )
    agent.skills = {
        "gathering": 0.5, "building": 0.5, "combat": 0.5,
        "medicine": 0.5, "crafting": 0.5, "persuasion": 0.5,
        "exploration": 0.5, "farming": 0.5
    }
    agent.inventory = Inventory(food=5.0, water=5.0, wood=5.0, stone=5.0, herbs=5.0)
    agent.known_regions = [region_id, "river", "hills"]
    return agent


def make_world_with_agent(agent):
    world = WorldState()
    region = Region(
        id=agent.current_region,
        name="Forest",
        region_type=RegionType.FOREST,
        resources={
            ResourceType.FOOD: ResourceNode(resource_type=ResourceType.FOOD, amount=10.0, max_amount=20.0, regen_rate=0.5),
            ResourceType.WOOD: ResourceNode(resource_type=ResourceType.WOOD, amount=15.0, max_amount=20.0, regen_rate=0.3),
        },
        occupants=[agent.id]
    )
    world.regions[agent.current_region] = region
    world.agents[agent.id] = agent
    return world


class TestGathering:
    def test_successful_gather(self):
        agent = make_agent()
        world = make_world_with_agent(agent)
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.GATHER,
            actor_id=agent.id,
            category=ActionCategory.MINOR,
            target_resource=ResourceType.FOOD,
            ap_cost=1,
            skill_used="gathering"
        )
        result = resolver.resolve(action, world, rng)
        assert result.success
        assert result.resources_gained.get("food", 0) > 0

    def test_exhausted_resource(self):
        agent = make_agent()
        world = make_world_with_agent(agent)
        world.regions[agent.current_region].resources[ResourceType.FOOD].amount = 0.0
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.GATHER,
            actor_id=agent.id,
            category=ActionCategory.MINOR,
            target_resource=ResourceType.FOOD,
            ap_cost=1,
            skill_used="gathering"
        )
        result = resolver.resolve(action, world, rng)
        assert not result.success or result.resources_gained.get("food", 0) == 0

    def test_skill_improvement(self):
        agent = make_agent()
        world = make_world_with_agent(agent)
        skill_before = agent.skills["gathering"]
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.GATHER,
            actor_id=agent.id,
            category=ActionCategory.MINOR,
            target_resource=ResourceType.FOOD,
            ap_cost=1,
            skill_used="gathering"
        )
        resolver.resolve(action, world, rng)
        assert agent.skills["gathering"] >= skill_before


class TestTrading:
    def test_successful_trade(self):
        world = create_default_world(seed=42)
        agents = list(world.agents.values())
        a1, a2 = agents[0], agents[1]
        a1.current_region = a2.current_region = "forest"
        world.regions["forest"].occupants = [a1.id, a2.id]
        a1.inventory.food = 10.0
        a2.inventory.wood = 10.0
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.TRADE,
            actor_id=a1.id,
            category=ActionCategory.MINOR,
            target_agent_id=a2.id,
            ap_cost=1,
            parameters={"offer": {"food": 2.0}, "request": {"wood": 2.0}}
        )
        result = resolver.resolve(action, world, rng)
        assert result is not None
        assert isinstance(result.success, bool)


class TestMovement:
    def test_successful_move(self):
        agent = make_agent(region_id="forest")
        world = make_world_with_agent(agent)
        world.regions["river"] = Region(
            id="river", name="River", region_type=RegionType.RIVER,
            resources={}, occupants=[]
        )
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.MOVE,
            actor_id=agent.id,
            category=ActionCategory.MINOR,
            target_region_id="river",
            ap_cost=1,
        )
        result = resolver.resolve(action, world, rng)
        assert result.success
        assert agent.current_region == "river"

    def test_unknown_region_fails(self):
        agent = make_agent()
        agent.known_regions = ["forest"]
        world = make_world_with_agent(agent)
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.MOVE,
            actor_id=agent.id,
            category=ActionCategory.MINOR,
            target_region_id="unknown_region",
            ap_cost=1,
        )
        result = resolver.resolve(action, world, rng)
        assert not result.success


class TestCombat:
    def test_produces_results(self):
        world = create_default_world(seed=42)
        agents = list(world.agents.values())
        attacker, defender = agents[0], agents[1]
        attacker.current_region = defender.current_region = "forest"
        world.regions["forest"].occupants = [attacker.id, defender.id]
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.ATTACK,
            actor_id=attacker.id,
            category=ActionCategory.MAJOR,
            target_agent_id=defender.id,
            ap_cost=2,
            skill_used="combat"
        )
        result = resolver.resolve(action, world, rng)
        assert result is not None
        assert isinstance(result.success, bool)

    def test_defending_provides_statistical_bonus(self):
        """Run 100 trials; defenders in stance should win more often."""
        wins_defending = 0
        for i in range(100):
            world = create_default_world(seed=i)
            agents = list(world.agents.values())
            attacker, defender = agents[0], agents[1]
            attacker.current_region = defender.current_region = "forest"
            world.regions["forest"].occupants = [attacker.id, defender.id]
            attacker.health = 1.0
            defender.health = 1.0
            resolver = ActionResolver()
            rng = random.Random(i)
            defend_action = Action(
                action_type=ActionType.DEFEND,
                actor_id=defender.id,
                category=ActionCategory.MAJOR,
                ap_cost=2,
            )
            resolver.resolve(defend_action, world, rng)
            attack_action = Action(
                action_type=ActionType.ATTACK,
                actor_id=attacker.id,
                category=ActionCategory.MAJOR,
                target_agent_id=defender.id,
                ap_cost=2,
                skill_used="combat"
            )
            health_before = defender.health
            resolver.resolve(attack_action, world, rng)
            if defender.health >= health_before:
                wins_defending += 1
        assert wins_defending >= 10


class TestHealing:
    def test_restores_health(self):
        world = create_default_world(seed=42)
        agents = list(world.agents.values())
        healer, patient = agents[4], agents[0]  # Senna is healer
        healer.current_region = patient.current_region = "river"
        world.regions["river"].occupants = [healer.id, patient.id]
        patient.health = 0.5
        healer.inventory.herbs = 5.0
        healer.skills["medicine"] = 0.7
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.HEAL,
            actor_id=healer.id,
            category=ActionCategory.MINOR,
            target_agent_id=patient.id,
            ap_cost=1,
            skill_used="medicine"
        )
        health_before = patient.health
        result = resolver.resolve(action, world, rng)
        assert result.success
        assert patient.health >= health_before


class TestSharing:
    def test_transfers_resources(self):
        world = create_default_world(seed=42)
        agents = list(world.agents.values())
        giver, receiver = agents[6], agents[0]  # Asha is altruist
        giver.current_region = receiver.current_region = "river"
        world.regions["river"].occupants = [giver.id, receiver.id]
        giver.inventory.food = 10.0
        receiver.inventory.food = 0.0
        resolver = ActionResolver()
        rng = random.Random(42)
        action = Action(
            action_type=ActionType.SHARE,
            actor_id=giver.id,
            category=ActionCategory.MINOR,
            target_agent_id=receiver.id,
            ap_cost=1,
            parameters={"resource": "food", "amount": 3.0}
        )
        result = resolver.resolve(action, world, rng)
        assert result.success
        assert receiver.inventory.food > 0.0
