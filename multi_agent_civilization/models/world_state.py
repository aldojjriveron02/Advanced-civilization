from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Season(Enum):
    SPRING = "SPRING"
    SUMMER = "SUMMER"
    AUTUMN = "AUTUMN"
    WINTER = "WINTER"


class WeatherType(Enum):
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    STORM = "STORM"
    DROUGHT = "DROUGHT"
    SNOW = "SNOW"
    FOG = "FOG"


class RegionType(Enum):
    FOREST = "FOREST"
    RIVER = "RIVER"
    HILLS = "HILLS"
    DESERT = "DESERT"
    COAST = "COAST"
    VOLCANIC = "VOLCANIC"
    RUINS = "RUINS"


class ResourceType(Enum):
    FOOD = "FOOD"
    WATER = "WATER"
    WOOD = "WOOD"
    STONE = "STONE"
    HERBS = "HERBS"
    METAL = "METAL"
    KNOWLEDGE = "KNOWLEDGE"


@dataclass
class ResourceNode:
    resource_type: ResourceType
    amount: float
    max_amount: float
    regen_rate: float
    depletion_modifier: float = 1.0
    overharvest_threshold: float = 0.3

    def harvest(self, amount: float) -> float:
        """Return actual amount taken; apply depletion modifier if below threshold."""
        ratio = self.amount / self.max_amount if self.max_amount > 0 else 0.0
        if ratio < self.overharvest_threshold:
            effective = min(amount * self.depletion_modifier, self.amount)
        else:
            effective = min(amount, self.amount)
        self.amount = max(0.0, self.amount - effective)
        return effective

    def regenerate(self, season_mod: float, weather_mod: float) -> None:
        """Regenerate resource up to max_amount."""
        gain = self.regen_rate * season_mod * weather_mod
        self.amount = min(self.max_amount, self.amount + gain)


@dataclass
class Structure:
    id: str
    name: str
    region_id: str
    builder_id: str
    durability: float = 1.0
    capacity: int = 5
    defense_bonus: float = 0.0
    storage_bonus: float = 0.0
    construction_progress: float = 0.0
    is_complete: bool = False
    turns_to_build: int = 2


@dataclass
class Region:
    id: str
    name: str
    region_type: RegionType
    resources: Dict[ResourceType, ResourceNode] = field(default_factory=dict)
    structures: List[Structure] = field(default_factory=list)
    occupants: List[str] = field(default_factory=list)
    environmental_scars: List[str] = field(default_factory=list)
    defensibility: float = 0.5
    visibility: float = 0.8
    flood_risk: float = 0.1
    fire_risk: float = 0.1
    is_discovered: bool = True


@dataclass
class WorldState:
    turn: int = 0
    season: Season = Season.SPRING
    weather: WeatherType = WeatherType.CLEAR
    regions: Dict[str, Region] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)
    factions: Dict[str, Any] = field(default_factory=dict)
    event_log: List[str] = field(default_factory=list)
    turn_history: List[Dict] = field(default_factory=list)

    @property
    def season_resource_modifier(self) -> float:
        return {
            Season.SPRING: 1.3,
            Season.SUMMER: 1.5,
            Season.AUTUMN: 1.0,
            Season.WINTER: 0.3,
        }[self.season]

    @property
    def weather_resource_modifier(self) -> float:
        return {
            WeatherType.CLEAR: 1.0,
            WeatherType.RAIN: 1.2,
            WeatherType.STORM: 0.5,
            WeatherType.DROUGHT: 0.3,
            WeatherType.SNOW: 0.2,
            WeatherType.FOG: 0.9,
        }[self.weather]

    def advance_season(self) -> None:
        """Advance season every 8 turns."""
        if self.turn > 0 and self.turn % 8 == 0:
            seasons = list(Season)
            idx = seasons.index(self.season)
            self.season = seasons[(idx + 1) % len(seasons)]
