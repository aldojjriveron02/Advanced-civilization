from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from models.faction import Faction, FactionNorm
from models.agent_state import Emotion, EmotionType, MemoryEntry, Relationship

if TYPE_CHECKING:
    from models.world_state import WorldState
    from models.actions import Action


def _default_relationship(respect: float = 0.5) -> Relationship:
    """Return a default Relationship with the given respect value."""
    return Relationship(respect=respect)


class FactionManager:
    def detect_proto_factions(self, world_state: "WorldState") -> List[Faction]:
        """BFS cluster detection for new factions."""
        agents = {aid: a for aid, a in world_state.agents.items() if a.is_alive}
        agent_ids = list(agents.keys())

        # Build adjacency: mutual trust>0.6 AND affinity>0.3
        adjacency: Dict[str, List[str]] = {aid: [] for aid in agent_ids}
        for i, a1 in enumerate(agent_ids):
            for a2 in agent_ids[i + 1:]:
                agent1 = agents[a1]
                agent2 = agents[a2]
                rel12 = agent1.relationships.get(a2)
                rel21 = agent2.relationships.get(a1)
                if (rel12 and rel12.trust > 0.6 and rel12.affinity > 0.3 and
                        rel21 and rel21.trust > 0.6 and rel21.affinity > 0.3):
                    adjacency[a1].append(a2)
                    adjacency[a2].append(a1)

        # BFS to find connected components
        visited = set()
        components = []
        for aid in agent_ids:
            if aid in visited:
                continue
            component = []
            queue = [aid]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(adjacency[node])
            components.append(component)

        new_factions = []
        for comp in components:
            if len(comp) < 3:
                continue
            # Check >60% of pairs are connected
            pair_count = len(comp) * (len(comp) - 1) // 2
            connected_pairs = sum(
                1 for i, a1 in enumerate(comp)
                for a2 in comp[i + 1:]
                if a2 in adjacency[a1]
            )
            if pair_count == 0 or connected_pairs / pair_count < 0.6:
                continue

            # Skip if members already in same faction
            existing_factions = set(
                agents[m].faction_id for m in comp if agents[m].faction_id
            )
            if len(existing_factions) == 1 and existing_factions:
                continue

            leader_id = self.elect_leader_from_members(comp, world_state)
            faction_id = str(uuid.uuid4())[:8]
            faction = Faction(
                id=faction_id,
                name=f"Faction_{faction_id[:4]}",
                founder_id=leader_id,
                members=list(comp),
                leader_id=leader_id,
                formed_turn=world_state.turn,
            )
            new_factions.append(faction)

        return new_factions

    def elect_leader_from_members(self, members: List[str], world_state: "WorldState") -> str:
        scores = {}
        for m in members:
            score = 0.0
            for other in members:
                if other == m:
                    continue
                agent = world_state.agents.get(other)
                if agent:
                    rel = agent.relationships.get(m)
                    if rel:
                        score += rel.respect
            scores[m] = score
        return max(scores, key=lambda x: scores[x]) if scores else members[0]

    def check_viability(self, world_state: "WorldState") -> None:
        to_dissolve = []
        for fid, faction in world_state.factions.items():
            # Remove dead members
            faction.members = [m for m in faction.members if world_state.agents.get(m, None) and world_state.agents[m].is_alive]
            faction.update_cohesion(world_state)

            if not faction.is_viable:
                to_dissolve.append(fid)

        for fid in to_dissolve:
            faction = world_state.factions[fid]
            for mid in faction.members:
                agent = world_state.agents.get(mid)
                if agent:
                    agent.faction_id = None
                    agent.emotions.append(Emotion(EmotionType.GRIEF, 0.5, f"faction_{fid}_dissolved"))
            del world_state.factions[fid]

    def elect_leader(self, faction: Faction, world_state: "WorldState") -> str:
        if not faction.members:
            return ""
        return self.elect_leader_from_members(faction.members, world_state)

    def check_leadership_challenge(self, faction: Faction, world_state: "WorldState") -> None:
        if len(faction.members) < 3:
            return

        leader = world_state.agents.get(faction.leader_id)
        if not leader:
            return

        leader_respect = sum(
            world_state.agents[m].relationships.get(faction.leader_id, _default_relationship()).respect
            for m in faction.members if m != faction.leader_id and world_state.agents.get(m)
        ) / max(1, len(faction.members) - 1)

        for challenger_id in faction.members:
            if challenger_id == faction.leader_id:
                continue
            challenger = world_state.agents.get(challenger_id)
            if not challenger or challenger.personality.aggression <= 0.4:
                continue

            challenger_respect = sum(
                world_state.agents[m].relationships.get(challenger_id, _default_relationship()).respect
                for m in faction.members if m != challenger_id and world_state.agents.get(m)
            ) / max(1, len(faction.members) - 1)

            if challenger_respect > leader_respect * 1.3:
                # Challenge: count supporters
                supporters = sum(
                    1 for m in faction.members
                    if m != faction.leader_id and
                    world_state.agents.get(m) and
                    world_state.agents[m].relationships.get(challenger_id, _default_relationship(0.0)).respect >
                    world_state.agents[m].relationships.get(faction.leader_id, _default_relationship(0.0)).respect
                )
                if supporters > len(faction.members) // 2:
                    faction.leader_id = challenger_id
                break

    def check_schism(self, faction: Faction, world_state: "WorldState") -> Optional[Tuple[Faction, Faction]]:
        if len(faction.members) < 4:
            return None

        hostile_pairs = 0
        total_pairs = 0
        for i, m1 in enumerate(faction.members):
            for m2 in faction.members[i + 1:]:
                total_pairs += 1
                a1 = world_state.agents.get(m1)
                if a1:
                    rel = a1.relationships.get(m2)
                    if rel and (rel.affinity < -0.2 or rel.trust < 0.3):
                        hostile_pairs += 1

        if total_pairs == 0 or hostile_pairs / total_pairs < 0.3:
            return None

        # Split into loyalists (support leader) and dissenters
        loyalists = [faction.leader_id]
        dissenters = []
        for m in faction.members:
            if m == faction.leader_id:
                continue
            a = world_state.agents.get(m)
            if a:
                leader_rel = a.relationships.get(faction.leader_id)
                if leader_rel and leader_rel.trust > 0.4:
                    loyalists.append(m)
                else:
                    dissenters.append(m)

        if len(loyalists) < 2 or len(dissenters) < 2:
            return None

        fid1 = str(uuid.uuid4())[:8]
        fid2 = str(uuid.uuid4())[:8]
        f1 = Faction(id=fid1, name=f"Loyal_{fid1[:4]}", founder_id=faction.leader_id,
                     members=loyalists, leader_id=faction.leader_id, formed_turn=world_state.turn)
        f2 = Faction(id=fid2, name=f"Dissent_{fid2[:4]}", founder_id=dissenters[0],
                     members=dissenters, leader_id=dissenters[0], formed_turn=world_state.turn)
        return f1, f2

    def check_merger(self, faction1: Faction, faction2: Faction, world_state: "WorldState") -> Optional[Faction]:
        if faction1.size + faction2.size > 7:
            return None

        cross_trusts = []
        for m1 in faction1.members:
            agent = world_state.agents.get(m1)
            if not agent:
                continue
            for m2 in faction2.members:
                rel = agent.relationships.get(m2)
                if rel:
                    cross_trusts.append(rel.trust)

        if not cross_trusts:
            return None

        avg_trust = sum(cross_trusts) / len(cross_trusts)
        if avg_trust <= 0.6:
            return None

        fid = str(uuid.uuid4())[:8]
        merged_members = faction1.members + faction2.members
        leader_id = self.elect_leader_from_members(merged_members, world_state)
        merged = Faction(
            id=fid,
            name=f"Merged_{fid[:4]}",
            founder_id=leader_id,
            members=merged_members,
            leader_id=leader_id,
            formed_turn=world_state.turn,
        )
        return merged

    def check_norm_compliance(self, agent: Any, action: "Action", faction: Faction, world_state: "WorldState") -> None:
        from models.agent_state import ActionType
        for norm in faction.norms:
            if "no stealing" in norm.description.lower():
                if hasattr(action, 'action_type') and action.action_type == ActionType.STEAL:
                    norm.violation_count += 1
                    for mid in faction.members:
                        if mid != agent.id:
                            member = world_state.agents.get(mid)
                            if member:
                                rel = member.get_relationship(agent.id)
                                rel.trust = max(0.0, rel.trust - 0.1)
                else:
                    norm.compliance_count += 1

    def request_join(self, agent_id: str, faction: Faction, world_state: "WorldState") -> bool:
        if not faction.members:
            faction.add_member(agent_id)
            return True

        votes_yes = 0
        votes_no = 0
        applicant = world_state.agents.get(agent_id)
        for mid in faction.members:
            member = world_state.agents.get(mid)
            if not member:
                continue
            rel = member.relationships.get(agent_id)
            trust = rel.trust if rel else 0.5
            if trust > 0.5:
                votes_yes += 1
            else:
                votes_no += 1

        if votes_yes > votes_no:
            faction.add_member(agent_id)
            if applicant:
                applicant.faction_id = faction.id
            return True
        return False

    def request_leave(self, agent_id: str, faction: Faction, world_state: "WorldState") -> None:
        faction.remove_member(agent_id)
        agent = world_state.agents.get(agent_id)
        if agent:
            agent.faction_id = None

        for mid in faction.members:
            member = world_state.agents.get(mid)
            if member:
                rel = member.get_relationship(agent_id)
                rel.trust = max(0.0, rel.trust - 0.05)
                rel.affinity = max(-1.0, rel.affinity - 0.05)

    def update_all(self, world_state: "WorldState") -> None:
        # Check viability first
        self.check_viability(world_state)

        # Detect new proto-factions
        new_factions = self.detect_proto_factions(world_state)
        for faction in new_factions:
            # Only add if genuinely new (members not already all in same faction)
            member_factions = set(
                world_state.agents[m].faction_id
                for m in faction.members
                if world_state.agents.get(m) and world_state.agents[m].faction_id
            )
            if len(member_factions) == 1 and member_factions:
                continue
            world_state.factions[faction.id] = faction
            for mid in faction.members:
                agent = world_state.agents.get(mid)
                if agent:
                    agent.faction_id = faction.id

        # Update existing factions
        faction_ids = list(world_state.factions.keys())
        for fid in faction_ids:
            if fid not in world_state.factions:
                continue
            faction = world_state.factions[fid]
            self.check_leadership_challenge(faction, world_state)

        # Check schisms
        for fid in list(world_state.factions.keys()):
            if fid not in world_state.factions:
                continue
            faction = world_state.factions[fid]
            result = self.check_schism(faction, world_state)
            if result:
                f1, f2 = result
                del world_state.factions[fid]
                world_state.factions[f1.id] = f1
                world_state.factions[f2.id] = f2
                for mid in f1.members:
                    a = world_state.agents.get(mid)
                    if a:
                        a.faction_id = f1.id
                for mid in f2.members:
                    a = world_state.agents.get(mid)
                    if a:
                        a.faction_id = f2.id

        # Check mergers
        faction_list = list(world_state.factions.values())
        for i in range(len(faction_list)):
            for j in range(i + 1, len(faction_list)):
                f1 = faction_list[i]
                f2 = faction_list[j]
                if f1.id not in world_state.factions or f2.id not in world_state.factions:
                    continue
                merged = self.check_merger(f1, f2, world_state)
                if merged:
                    del world_state.factions[f1.id]
                    del world_state.factions[f2.id]
                    world_state.factions[merged.id] = merged
                    for mid in merged.members:
                        a = world_state.agents.get(mid)
                        if a:
                            a.faction_id = merged.id
                    break
