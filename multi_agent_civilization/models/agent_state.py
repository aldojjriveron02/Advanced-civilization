from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.world_state import ResourceType


class Archetype(Enum):
    BUILDER = "BUILDER"
    DIPLOMAT = "DIPLOMAT"
    OPPORTUNIST = "OPPORTUNIST"
    HEALER = "HEALER"
    EXPLORER = "EXPLORER"
    HOARDER = "HOARDER"
    INVENTOR = "INVENTOR"
    PROTECTOR = "PROTECTOR"
    ALTRUIST = "ALTRUIST"
    STRATEGIST = "STRATEGIST"


class ActionType(Enum):
    GATHER = "GATHER"
    BUILD = "BUILD"
    TRADE = "TRADE"
    EXPLORE = "EXPLORE"
    REST = "REST"
    CRAFT = "CRAFT"
    ATTACK = "ATTACK"
    DEFEND = "DEFEND"
    HEAL = "HEAL"
    TEACH = "TEACH"
    PERSUADE = "PERSUADE"
    SABOTAGE = "SABOTAGE"
    SHARE = "SHARE"
    STEAL = "STEAL"
    MOVE = "MOVE"
    OBSERVE = "OBSERVE"
    COMMUNICATE = "COMMUNICATE"


class EmotionType(Enum):
    FEAR = "FEAR"
    ANGER = "ANGER"
    GRATITUDE = "GRATITUDE"
    ENVY = "ENVY"
    PRIDE = "PRIDE"
    GRIEF = "GRIEF"
    HOPE = "HOPE"
    DISGUST = "DISGUST"
    LONELINESS = "LONELINESS"


@dataclass
class Emotion:
    emotion_type: EmotionType
    intensity: float
    source: str
    decay_rate: float = 0.1

    def decay(self) -> bool:
        """Reduce intensity by decay_rate. Returns True if expired."""
        self.intensity = max(0.0, self.intensity - self.decay_rate)
        return self.intensity <= 0.0


@dataclass
class Relationship:
    trust: float = 0.5
    affinity: float = 0.0
    dependence: float = 0.0
    fear: float = 0.0
    respect: float = 0.5
    debt: float = 0.0

    def update_after_cooperation(self, amount: float = 0.1) -> None:
        self.trust = min(1.0, self.trust + amount)
        self.affinity = min(1.0, self.affinity + amount * 0.5)
        self.respect = min(1.0, self.respect + amount * 0.3)

    def update_after_betrayal(self) -> None:
        impact = self.trust * 0.5 + 0.1
        self.trust = max(0.0, self.trust - impact)
        self.affinity = max(-1.0, self.affinity - impact * 0.7)
        self.fear = min(1.0, self.fear + impact * 0.3)
        self.respect = max(0.0, self.respect - impact * 0.4)

    def decay(self, rate: float = 0.01) -> None:
        """Decay toward neutral values per turn."""
        self.trust += (0.5 - self.trust) * rate
        self.affinity += (0.0 - self.affinity) * rate
        self.respect += (0.5 - self.respect) * rate
        self.fear = max(0.0, self.fear - rate * 0.5)


@dataclass
class MemoryEntry:
    turn: int
    category: str
    content: str
    importance: float
    related_agents: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    verified: bool = False


@dataclass
class Belief:
    content: str
    confidence: float
    source: str
    falsifiability: float = 0.5

    def reinforce(self, amount: float = 0.1) -> None:
        self.confidence = min(1.0, self.confidence + amount)

    def contradict(self, amount: float = 0.1) -> None:
        self.confidence = max(0.0, self.confidence - amount)

    @property
    def is_superstition(self) -> bool:
        return self.falsifiability < 0.2 and self.confidence > 0.7


@dataclass
class Inventory:
    food: float = 0.0
    water: float = 0.0
    wood: float = 0.0
    stone: float = 0.0
    herbs: float = 0.0
    metal: float = 0.0
    max_capacity: float = 20.0

    def add(self, resource_type: Any, amount: float) -> float:
        """Add resource, respecting capacity. Returns amount actually added."""
        from models.world_state import ResourceType
        current = self.get(resource_type)
        can_add = min(amount, self.remaining_capacity)
        self.set(resource_type, current + can_add)
        return can_add

    def remove(self, resource_type: Any, amount: float) -> float:
        """Remove resource. Returns amount actually removed."""
        current = self.get(resource_type)
        actually_removed = min(amount, current)
        self.set(resource_type, current - actually_removed)
        return actually_removed

    @property
    def total(self) -> float:
        return self.food + self.water + self.wood + self.stone + self.herbs + self.metal

    @property
    def remaining_capacity(self) -> float:
        return max(0.0, self.max_capacity - self.total)

    def get(self, resource_type: Any) -> float:
        from models.world_state import ResourceType
        mapping = {
            ResourceType.FOOD: "food",
            ResourceType.WATER: "water",
            ResourceType.WOOD: "wood",
            ResourceType.STONE: "stone",
            ResourceType.HERBS: "herbs",
            ResourceType.METAL: "metal",
        }
        attr = mapping.get(resource_type, str(resource_type).lower().split(".")[-1].lower())
        return getattr(self, attr, 0.0)

    def set(self, resource_type: Any, value: float) -> None:
        from models.world_state import ResourceType
        mapping = {
            ResourceType.FOOD: "food",
            ResourceType.WATER: "water",
            ResourceType.WOOD: "wood",
            ResourceType.STONE: "stone",
            ResourceType.HERBS: "herbs",
            ResourceType.METAL: "metal",
        }
        attr = mapping.get(resource_type, str(resource_type).lower().split(".")[-1].lower())
        if hasattr(self, attr):
            setattr(self, attr, max(0.0, value))


@dataclass
class Needs:
    hunger: float = 0.0
    thirst: float = 0.0
    safety: float = 0.0
    belonging: float = 0.0
    influence: float = 0.0
    rest: float = 0.0

    CRITICAL: float = field(default=0.85, init=False, repr=False, compare=False)
    DANGER: float = field(default=0.7, init=False, repr=False, compare=False)
    COMFORT: float = field(default=0.3, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'CRITICAL', 0.85)
        object.__setattr__(self, 'DANGER', 0.7)
        object.__setattr__(self, 'COMFORT', 0.3)

    def tick(self) -> None:
        self.hunger = min(1.0, self.hunger + 0.08)
        self.thirst = min(1.0, self.thirst + 0.10)
        self.rest = min(1.0, self.rest + 0.05)

    def priority_order(self) -> List[tuple]:
        needs_list = [
            ('hunger', self.hunger),
            ('thirst', self.thirst),
            ('safety', self.safety),
            ('belonging', self.belonging),
            ('influence', self.influence),
            ('rest', self.rest),
        ]
        return sorted(needs_list, key=lambda x: x[1], reverse=True)

    @property
    def is_desperate(self) -> bool:
        return any(v >= 0.85 for v in [
            self.hunger, self.thirst, self.safety,
            self.belonging, self.influence, self.rest
        ])

    def auto_consume(self, inventory: Inventory) -> None:
        from models.world_state import ResourceType
        if self.hunger > 0.5 and inventory.food > 0:
            amount = min(1.0, inventory.food)
            inventory.remove(ResourceType.FOOD, amount)
            self.hunger = max(0.0, self.hunger - amount * 0.4)
        if self.thirst > 0.5 and inventory.water > 0:
            amount = min(1.0, inventory.water)
            inventory.remove(ResourceType.WATER, amount)
            self.thirst = max(0.0, self.thirst - amount * 0.5)


@dataclass
class PersonalityTraits:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    greed: float = 0.5
    honesty: float = 0.5
    loyalty: float = 0.5
    aggression: float = 0.5
    curiosity: float = 0.5


ARCHETYPE_PRESETS: Dict[Archetype, Dict[str, float]] = {
    Archetype.BUILDER: {
        "conscientiousness": 0.9,
        "agreeableness": 0.7,
        "greed": 0.2,
        "aggression": 0.2,
        "curiosity": 0.5,
    },
    Archetype.DIPLOMAT: {
        "extraversion": 0.8,
        "agreeableness": 0.9,
        "honesty": 0.7,
        "aggression": 0.1,
        "loyalty": 0.7,
    },
    Archetype.OPPORTUNIST: {
        "greed": 0.8,
        "honesty": 0.2,
        "aggression": 0.5,
        "openness": 0.7,
        "conscientiousness": 0.3,
    },
    Archetype.HEALER: {
        "agreeableness": 0.9,
        "conscientiousness": 0.7,
        "greed": 0.1,
        "aggression": 0.1,
        "honesty": 0.8,
    },
    Archetype.EXPLORER: {
        "openness": 0.9,
        "curiosity": 0.9,
        "extraversion": 0.6,
        "conscientiousness": 0.4,
        "greed": 0.3,
    },
    Archetype.HOARDER: {
        "greed": 0.9,
        "conscientiousness": 0.8,
        "agreeableness": 0.2,
        "loyalty": 0.3,
        "openness": 0.3,
    },
    Archetype.INVENTOR: {
        "openness": 0.9,
        "curiosity": 0.9,
        "conscientiousness": 0.7,
        "extraversion": 0.4,
        "greed": 0.3,
    },
    Archetype.PROTECTOR: {
        "aggression": 0.6,
        "loyalty": 0.9,
        "agreeableness": 0.7,
        "conscientiousness": 0.8,
        "greed": 0.2,
    },
    Archetype.ALTRUIST: {
        "agreeableness": 0.95,
        "greed": 0.05,
        "honesty": 0.9,
        "loyalty": 0.8,
        "extraversion": 0.6,
    },
    Archetype.STRATEGIST: {
        "openness": 0.7,
        "conscientiousness": 0.9,
        "aggression": 0.4,
        "greed": 0.5,
        "curiosity": 0.7,
    },
}


@dataclass
class Agent:
    id: str
    name: str
    archetype: Archetype
    personality: PersonalityTraits
    needs: Needs = field(default_factory=Needs)
    inventory: Inventory = field(default_factory=Inventory)
    emotions: List[Emotion] = field(default_factory=list)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    memories: List[MemoryEntry] = field(default_factory=list)
    beliefs: List[Belief] = field(default_factory=list)
    known_regions: List[str] = field(default_factory=list)
    current_region: str = ""
    health: float = 1.0
    action_points: int = 3
    is_alive: bool = True
    faction_id: Optional[str] = None
    skills: Dict[str, float] = field(default_factory=dict)
    pending_action: Optional[Any] = None
    project: Optional[Any] = None

    @property
    def cognitive_bandwidth(self) -> int:
        base = 5 if self.archetype == Archetype.STRATEGIST else 3
        if self.needs.is_desperate:
            critical_count = sum(
                1 for v in [
                    self.needs.hunger, self.needs.thirst, self.needs.safety,
                    self.needs.rest
                ] if v >= 0.85
            )
            base -= min(critical_count, base - 1)
        dom = self.dominant_emotion
        if dom and dom.intensity > 0.7:
            base = max(1, base - 1)
        return max(1, base)

    @property
    def dominant_emotion(self) -> Optional[Emotion]:
        if not self.emotions:
            return None
        return max(self.emotions, key=lambda e: e.intensity)

    @property
    def starvation_damage(self) -> float:
        damage = 0.0
        if self.needs.hunger >= 0.95:
            damage += 0.05
        if self.needs.thirst >= 0.95:
            damage += 0.08
        return damage

    def add_memory(self, entry: MemoryEntry) -> None:
        self.memories.append(entry)
        if len(self.memories) > 200:
            self.memories.sort(key=lambda m: m.importance)
            self.memories = self.memories[len(self.memories) - 200:]

    def get_relationship(self, agent_id: str) -> Relationship:
        if agent_id not in self.relationships:
            self.relationships[agent_id] = Relationship()
        return self.relationships[agent_id]
