from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from models.agent_state import ActionType


class ActionCategory(Enum):
    FREE = "FREE"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    FULL_TURN = "FULL_TURN"


ACTION_COSTS: Dict[ActionType, Tuple[ActionCategory, int]] = {
    ActionType.OBSERVE: (ActionCategory.FREE, 0),
    ActionType.COMMUNICATE: (ActionCategory.FREE, 0),
    ActionType.GATHER: (ActionCategory.MINOR, 1),
    ActionType.TRADE: (ActionCategory.MINOR, 1),
    ActionType.MOVE: (ActionCategory.MINOR, 1),
    ActionType.SHARE: (ActionCategory.MINOR, 1),
    ActionType.REST: (ActionCategory.MINOR, 1),
    ActionType.HEAL: (ActionCategory.MINOR, 1),
    ActionType.BUILD: (ActionCategory.MAJOR, 2),
    ActionType.EXPLORE: (ActionCategory.MAJOR, 2),
    ActionType.ATTACK: (ActionCategory.MAJOR, 2),
    ActionType.DEFEND: (ActionCategory.MAJOR, 2),
    ActionType.CRAFT: (ActionCategory.MAJOR, 2),
    ActionType.TEACH: (ActionCategory.MAJOR, 2),
    ActionType.PERSUADE: (ActionCategory.MAJOR, 2),
    ActionType.SABOTAGE: (ActionCategory.MAJOR, 2),
    ActionType.STEAL: (ActionCategory.MAJOR, 2),
}


@dataclass
class Action:
    action_type: ActionType
    actor_id: str
    category: ActionCategory
    target_agent_id: Optional[str] = None
    target_region_id: Optional[str] = None
    target_resource: Optional[object] = None
    parameters: Dict = field(default_factory=dict)
    ap_cost: int = 1
    skill_used: str = ""
    success_probability: float = 0.8
    is_continuation: bool = False
    project_id: Optional[str] = None


@dataclass
class ActionResult:
    success: bool
    outcome: str
    resources_gained: Dict[str, float] = field(default_factory=dict)
    resources_lost: Dict[str, float] = field(default_factory=dict)
    relationship_changes: Dict[str, Dict] = field(default_factory=dict)
    triggered_events: List[str] = field(default_factory=list)
    witnesses: List[str] = field(default_factory=list)


@dataclass
class Project:
    id: str
    action_type: ActionType
    actor_id: str
    progress: float = 0.0
    required_progress: float = 2.0
    resource_requirements: Dict = field(default_factory=dict)
    collaborators: List[str] = field(default_factory=list)
    target_region_id: str = ""
    parameters: Dict = field(default_factory=dict)
