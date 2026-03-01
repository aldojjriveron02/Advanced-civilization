from __future__ import annotations
import random as _random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models.world_state import WeatherType, Season
from models.agent_state import ActionType, MemoryEntry
from models.actions import Action, ActionCategory, ACTION_COSTS
from systems.event_system import EventSystem
from systems.action_resolver import ActionResolver
from systems.faction_manager import FactionManager
from systems.communication_parser import CommunicationParser
from systems.innovation import InnovationSystem
from systems.memory_manager import MemoryManager
from analysis.metrics import compute_gini, compute_alliance_density

if TYPE_CHECKING:
    from models.world_state import WorldState
    from systems.llm_interface import LLMInterface


@dataclass
class TurnSnapshot:
    turn: int
    season: str
    weather: str
    alive_count: int
    gini: float
    alliance_density: float
    avg_health: float
    avg_hunger: float
    faction_count: int
    structure_count: int
    events: List[str]
    deaths: List[str]


@dataclass
class LogEntry:
    turn: int
    agent_id: str
    action_type: str
    outcome: str
    success: bool


# Markov weather transition tables
WEATHER_TRANSITIONS: Dict[WeatherType, Dict[WeatherType, float]] = {
    WeatherType.CLEAR: {WeatherType.CLEAR: 0.6, WeatherType.RAIN: 0.25, WeatherType.STORM: 0.05, WeatherType.DROUGHT: 0.1},
    WeatherType.RAIN: {WeatherType.CLEAR: 0.4, WeatherType.RAIN: 0.3, WeatherType.STORM: 0.2, WeatherType.DROUGHT: 0.1},
    WeatherType.STORM: {WeatherType.CLEAR: 0.3, WeatherType.RAIN: 0.3, WeatherType.STORM: 0.2, WeatherType.SNOW: 0.1, WeatherType.FOG: 0.1},
    WeatherType.DROUGHT: {WeatherType.CLEAR: 0.4, WeatherType.DROUGHT: 0.4, WeatherType.RAIN: 0.2},
    WeatherType.SNOW: {WeatherType.CLEAR: 0.3, WeatherType.SNOW: 0.4, WeatherType.RAIN: 0.2, WeatherType.FOG: 0.1},
    WeatherType.FOG: {WeatherType.CLEAR: 0.5, WeatherType.FOG: 0.3, WeatherType.RAIN: 0.2},
}


def _markov_weather(current: WeatherType, season: Season, rng: Any) -> WeatherType:
    transitions = dict(WEATHER_TRANSITIONS.get(current, {WeatherType.CLEAR: 1.0}))

    # Season overrides
    if season == Season.WINTER:
        snow_boost = 0.15
        transitions[WeatherType.SNOW] = transitions.get(WeatherType.SNOW, 0) + snow_boost
        for k in transitions:
            if k != WeatherType.SNOW:
                transitions[k] = max(0.0, transitions[k] - snow_boost / max(1, len(transitions) - 1))
    elif season == Season.SUMMER:
        drought_boost = 0.1
        transitions[WeatherType.DROUGHT] = transitions.get(WeatherType.DROUGHT, 0) + drought_boost
        for k in transitions:
            if k != WeatherType.DROUGHT:
                transitions[k] = max(0.0, transitions[k] - drought_boost / max(1, len(transitions) - 1))

    # Normalize
    total = sum(transitions.values())
    if total <= 0:
        return WeatherType.CLEAR
    normalized = {k: v / total for k, v in transitions.items()}

    roll = rng.random()
    cumulative = 0.0
    for weather, prob in normalized.items():
        cumulative += prob
        if roll <= cumulative:
            return weather
    return WeatherType.CLEAR


class TurnEngine:
    def __init__(self, world_state: "WorldState", llm_interface: "LLMInterface", seed: Optional[int] = None) -> None:
        self.world_state = world_state
        self.llm = llm_interface
        self.rng = _random.Random(seed)
        self.event_system = EventSystem()
        self.action_resolver = ActionResolver()
        self.faction_manager = FactionManager()
        self.communication_parser = CommunicationParser()
        self.innovation_system = InnovationSystem()
        self.memory_manager = MemoryManager()
        self.log: List[LogEntry] = []

    def execute_turn(self) -> TurnSnapshot:
        ws = self.world_state
        ws.turn += 1
        deaths_this_turn: List[str] = []
        events_this_turn: List[str] = []

        # --- Phase 1: Environment ---
        ws.advance_season()
        ws.weather = _markov_weather(ws.weather, ws.season, self.rng)

        s_mod = ws.season_resource_modifier
        w_mod = ws.weather_resource_modifier
        for region in ws.regions.values():
            for node in region.resources.values():
                node.regenerate(s_mod, w_mod)
            for struct in region.structures:
                struct.durability = max(0.0, struct.durability - 0.002)

        # --- Phase 2: Events ---
        triggered_events = self.event_system.roll_events(ws, self.rng)
        for ev in triggered_events:
            events_this_turn.append(ev["name"])

        # --- Phase 3: Needs tick ---
        for agent in ws.agents.values():
            if not agent.is_alive:
                continue
            agent.needs.tick()
            agent.emotions = [e for e in agent.emotions if not e.decay()]
            agent.action_points = 3
            agent.needs.auto_consume(agent.inventory)

        # --- Phase 4: Observation ---
        observations: Dict[str, str] = {}
        for aid, agent in ws.agents.items():
            if not agent.is_alive:
                continue
            observations[aid] = self._build_observation(agent, ws)

        # --- Phase 5 & 6: Action declaration + conversion ---
        agent_actions: Dict[str, List[Action]] = {}
        speeches: Dict[str, str] = {}

        for aid, agent in ws.agents.items():
            if not agent.is_alive:
                continue
            memory_summary = self.memory_manager.build_context_summary(agent, ws.turn)
            prompt = self._build_agent_prompt(agent, observations[aid], memory_summary, ws)
            response = self.llm.call(prompt)

            if response:
                parsed = self.llm.parse_response(response)
                actions = self._convert_actions(parsed.get("actions", []), agent, ws)
                speech = parsed.get("speech", "")
            else:
                actions = self._heuristic_actions(agent, ws)
                speech = ""

            agent_actions[aid] = actions
            speeches[aid] = speech

        # --- Phase 7: Conflict detection (passive) ---
        # (handled implicitly during resolution)

        # --- Phase 8: Action resolution ---
        for aid, actions in agent_actions.items():
            agent = ws.agents.get(aid)
            if not agent or not agent.is_alive:
                continue
            ap_spent = 0
            for action in actions:
                if ap_spent + action.ap_cost > agent.action_points:
                    break
                result = self.action_resolver.resolve(action, ws, self.rng)
                ap_spent += action.ap_cost
                self.log.append(LogEntry(
                    turn=ws.turn,
                    agent_id=aid,
                    action_type=action.action_type.value,
                    outcome=result.outcome,
                    success=result.success,
                ))
                # Concept discovery
                self.innovation_system.check_concept_discovery(agent, action.action_type.value, self.rng)

        # --- Phase 9: Norm compliance ---
        for aid, actions in agent_actions.items():
            agent = ws.agents.get(aid)
            if not agent or not agent.is_alive or not agent.faction_id:
                continue
            faction = ws.factions.get(agent.faction_id)
            if faction:
                for action in actions:
                    self.faction_manager.check_norm_compliance(agent, action, faction, ws)

        # --- Phase 10: Communication ---
        speakers_this_turn = []
        for aid, speech in speeches.items():
            if not speech:
                continue
            agent = ws.agents.get(aid)
            if not agent or not agent.is_alive:
                continue
            speakers_this_turn.append(aid)
            self.communication_parser.parse_message(agent, speech, ws, ws.turn, self.rng)

        self.communication_parser._process_strategic_silence(ws, speakers_this_turn)

        # --- Phase 11: Faction update ---
        self.faction_manager.update_all(ws)

        # --- Phase 12: Survival check ---
        for agent in ws.agents.values():
            if not agent.is_alive:
                continue
            dmg = agent.starvation_damage
            if dmg > 0:
                agent.health = max(0.0, agent.health - dmg)
            if agent.health <= 0:
                agent.is_alive = False
                region = ws.regions.get(agent.current_region)
                if region and agent.id in region.occupants:
                    region.occupants.remove(agent.id)
                if agent.faction_id and agent.faction_id in ws.factions:
                    ws.factions[agent.faction_id].remove_member(agent.id)
                deaths_this_turn.append(agent.id)
                ws.event_log.append(f"Turn {ws.turn}: {agent.name} has died.")

        # --- Phase 13: Relationship decay ---
        for agent in ws.agents.values():
            for rel in agent.relationships.values():
                rel.decay()

        # --- Phase 14: Metrics ---
        alive_agents = [a for a in ws.agents.values() if a.is_alive]
        alive_count = len(alive_agents)

        food_values = [a.inventory.food for a in alive_agents]
        gini = compute_gini(food_values)
        alliance_density = compute_alliance_density(ws)
        avg_health = sum(a.health for a in alive_agents) / max(1, alive_count)
        avg_hunger = sum(a.needs.hunger for a in alive_agents) / max(1, alive_count)
        structure_count = sum(len(r.structures) for r in ws.regions.values())

        return TurnSnapshot(
            turn=ws.turn,
            season=ws.season.value,
            weather=ws.weather.value,
            alive_count=alive_count,
            gini=gini,
            alliance_density=alliance_density,
            avg_health=avg_health,
            avg_hunger=avg_hunger,
            faction_count=len(ws.factions),
            structure_count=structure_count,
            events=events_this_turn,
            deaths=deaths_this_turn,
        )

    def _build_observation(self, agent: Any, ws: "WorldState") -> str:
        region = ws.regions.get(agent.current_region)
        if not region:
            return "You are in an unknown location."

        lines = [f"You are in {region.name} ({region.region_type.value})."]

        # Resources
        resources_visible = []
        for rt, node in region.resources.items():
            # Storm/fog reduces visibility
            if ws.weather in (WeatherType.STORM, WeatherType.FOG):
                if self.rng.random() < 0.3:
                    continue
            resources_visible.append(f"{rt.value}: {node.amount:.1f}")
        if resources_visible:
            lines.append("Resources: " + ", ".join(resources_visible))

        # Agents present
        visible_agents = []
        for oid in region.occupants:
            if oid == agent.id:
                continue
            other = ws.agents.get(oid)
            if not other or not other.is_alive:
                continue
            if ws.weather in (WeatherType.STORM, WeatherType.FOG) and self.rng.random() < 0.5:
                continue
            rel = agent.relationships.get(oid)
            trust_desc = f"trust={rel.trust:.1f}" if rel else "stranger"
            visible_agents.append(f"{other.name} ({other.archetype.value}, {trust_desc})")
        if visible_agents:
            lines.append("Others present: " + ", ".join(visible_agents))

        # Structures
        complete_structs = [s for s in region.structures if s.is_complete]
        if complete_structs:
            lines.append("Structures: " + ", ".join(s.name for s in complete_structs))

        lines.append(f"Season: {ws.season.value}, Weather: {ws.weather.value}")

        return "\n".join(lines)

    def _build_agent_prompt(self, agent: Any, observation: str, memory_summary: str, ws: "WorldState") -> str:
        lines = [
            "=== YOUR IDENTITY ===",
            f"Name: {agent.name}",
            f"Archetype: {agent.archetype.value}",
            f"Personality: openness={agent.personality.openness:.1f}, conscientiousness={agent.personality.conscientiousness:.1f}, "
            f"agreeableness={agent.personality.agreeableness:.1f}, aggression={agent.personality.aggression:.1f}, "
            f"honesty={agent.personality.honesty:.1f}",
            "",
            "=== YOUR STATE ===",
            f"Health: {agent.health:.2f}",
            f"Action Points: {agent.action_points}",
            f"Hunger: {agent.needs.hunger:.2f} | Thirst: {agent.needs.thirst:.2f} | Rest: {agent.needs.rest:.2f}",
            f"Inventory: food={agent.inventory.food:.1f}, water={agent.inventory.water:.1f}, "
            f"wood={agent.inventory.wood:.1f}, stone={agent.inventory.stone:.1f}, "
            f"herbs={agent.inventory.herbs:.1f}, metal={agent.inventory.metal:.1f}",
            "",
            "=== YOUR OBSERVATIONS ===",
            observation,
            "",
            "=== YOUR MEMORIES ===",
            memory_summary,
            "",
            "=== INSTRUCTIONS ===",
            "Choose actions for this turn. Available action types:",
            "GATHER (target_resource: FOOD/WATER/WOOD/STONE/HERBS/METAL)",
            "BUILD (parameters: {build_type: shelter/wall/storehouse/farm/workshop})",
            "TRADE (target_agent_id, parameters: {offer: {resource: amount}, request: {resource: amount}})",
            "EXPLORE, REST, HEAL (target_agent_id optional)",
            "MOVE (target_region_id: forest/river/hills/coast/caves)",
            "SHARE (target_agent_id, parameters: {resource: food, amount: 2.0})",
            "OBSERVE, COMMUNICATE (free actions)",
            "ATTACK/DEFEND (target_agent_id for ATTACK)",
            "CRAFT (parameters: {craft_type: rope/stone_tools/stone_axe/medicine_pouch/fishing_net})",
            "TEACH/PERSUADE (target_agent_id, parameters: {skill: gathering})",
            "",
            'Respond with JSON actions list: [{"action_type": "GATHER", "target_resource": "FOOD", "parameters": {}}]',
            "You have 3 action points. MINOR actions cost 1 AP, MAJOR actions cost 2 AP.",
        ]
        return "\n".join(lines)

    def _convert_actions(self, raw_actions: List[Dict], agent: Any, ws: "WorldState") -> List[Action]:
        actions = []
        ap_used = 0
        for raw in raw_actions:
            try:
                action_type_str = raw.get("action_type", "").upper()
                try:
                    action_type = ActionType[action_type_str]
                except KeyError:
                    continue

                category, ap_cost = ACTION_COSTS.get(action_type, (ActionCategory.MINOR, 1))
                if ap_used + ap_cost > agent.action_points:
                    break

                # Parse resource type
                target_resource = None
                resource_str = raw.get("target_resource", "")
                if resource_str:
                    from models.world_state import ResourceType
                    try:
                        target_resource = ResourceType[resource_str.upper()]
                    except KeyError:
                        pass

                action = Action(
                    action_type=action_type,
                    actor_id=agent.id,
                    category=category,
                    target_agent_id=raw.get("target_agent_id"),
                    target_region_id=raw.get("target_region_id"),
                    target_resource=target_resource,
                    parameters=raw.get("parameters", {}),
                    ap_cost=ap_cost,
                    skill_used=raw.get("skill_used", ""),
                )
                actions.append(action)
                ap_used += ap_cost
            except Exception:
                continue
        return actions

    def _heuristic_actions(self, agent: Any, ws: "WorldState") -> List[Action]:
        """Simple fallback when LLM is unavailable."""
        actions = []
        ap_budget = agent.action_points
        from models.world_state import ResourceType

        # Drink water if thirsty
        if agent.needs.thirst > 0.5 and ap_budget >= 1:
            region = ws.regions.get(agent.current_region)
            if region and ResourceType.WATER in region.resources and region.resources[ResourceType.WATER].amount > 0:
                actions.append(Action(
                    action_type=ActionType.GATHER,
                    actor_id=agent.id,
                    category=ActionCategory.MINOR,
                    target_resource=ResourceType.WATER,
                    ap_cost=1,
                ))
                ap_budget -= 1

        # Eat if hungry
        if agent.needs.hunger > 0.5 and ap_budget >= 1:
            region = ws.regions.get(agent.current_region)
            if region and ResourceType.FOOD in region.resources and region.resources[ResourceType.FOOD].amount > 0:
                actions.append(Action(
                    action_type=ActionType.GATHER,
                    actor_id=agent.id,
                    category=ActionCategory.MINOR,
                    target_resource=ResourceType.FOOD,
                    ap_cost=1,
                ))
                ap_budget -= 1

        # Rest if tired
        if agent.needs.rest > 0.6 and ap_budget >= 1:
            actions.append(Action(
                action_type=ActionType.REST,
                actor_id=agent.id,
                category=ActionCategory.MINOR,
                ap_cost=1,
            ))
            ap_budget -= 1

        # Explore if nothing else
        if not actions and ap_budget >= 2:
            actions.append(Action(
                action_type=ActionType.EXPLORE,
                actor_id=agent.id,
                category=ActionCategory.MAJOR,
                ap_cost=2,
            ))

        return actions
