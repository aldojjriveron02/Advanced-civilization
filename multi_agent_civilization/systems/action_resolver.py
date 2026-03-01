from __future__ import annotations
from typing import Any, TYPE_CHECKING

from models.actions import Action, ActionResult, ActionCategory, ACTION_COSTS, Project
from models.agent_state import ActionType, Emotion, EmotionType, MemoryEntry
from models.world_state import ResourceType, RegionType, Structure

if TYPE_CHECKING:
    from models.world_state import WorldState


class ActionResolver:
    def resolve(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        dispatch = {
            ActionType.GATHER: self._resolve_gather,
            ActionType.BUILD: self._resolve_build,
            ActionType.TRADE: self._resolve_trade,
            ActionType.EXPLORE: self._resolve_explore,
            ActionType.ATTACK: self._resolve_combat,
            ActionType.HEAL: self._resolve_heal,
            ActionType.MOVE: self._resolve_move,
            ActionType.SHARE: self._resolve_share,
            ActionType.REST: self._resolve_rest,
            ActionType.TEACH: self._resolve_teach,
            ActionType.CRAFT: self._resolve_craft,
            ActionType.STEAL: self._resolve_steal,
            ActionType.SABOTAGE: self._resolve_sabotage,
            ActionType.PERSUADE: self._resolve_persuade,
            ActionType.OBSERVE: self._resolve_observe,
            ActionType.DEFEND: self._resolve_defend,
            ActionType.COMMUNICATE: lambda a, ws, r: ActionResult(True, "Agent communicated."),
        }
        fn = dispatch.get(action.action_type)
        if fn:
            return fn(action, world_state, rng)
        return ActionResult(False, f"Unknown action type: {action.action_type}")

    # --- GATHER ---
    def _resolve_gather(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")
        region = world_state.regions.get(agent.current_region)
        if not region:
            return ActionResult(False, "Region not found.")

        resource_type = action.target_resource
        if resource_type is None:
            # Pick most available resource
            if not region.resources:
                return ActionResult(False, "No resources in region.")
            resource_type = max(region.resources, key=lambda rt: region.resources[rt].amount)

        node = region.resources.get(resource_type)
        if not node or node.amount <= 0:
            return ActionResult(False, f"No {resource_type.value} available.", resources_gained={})

        skill = agent.skills.get("gathering", 0.3)
        weather_mod = world_state.weather_resource_modifier
        yield_amount = 2.0 * (0.5 + skill) * weather_mod
        actual = node.harvest(yield_amount)

        agent.inventory.add(resource_type, actual)
        agent.skills["gathering"] = min(1.0, agent.skills.get("gathering", 0.3) + 0.01)

        return ActionResult(
            True,
            f"{agent.name} gathered {actual:.1f} {resource_type.value}.",
            resources_gained={resource_type.value.lower(): actual},
        )

    # --- BUILD ---
    def _resolve_build(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")
        region = world_state.regions.get(agent.current_region)
        if not region:
            return ActionResult(False, "Region not found.")

        build_configs = {
            "shelter": {"turns": 2, "resources": {"wood": 4.0}},
            "wall": {"turns": 3, "resources": {"stone": 6.0, "wood": 2.0}},
            "storehouse": {"turns": 3, "resources": {"wood": 6.0, "stone": 3.0}},
            "farm": {"turns": 2, "resources": {"wood": 2.0}},
            "workshop": {"turns": 4, "resources": {"wood": 5.0, "stone": 5.0}},
        }

        build_type = action.parameters.get("build_type", "shelter")
        config = build_configs.get(build_type, build_configs["shelter"])

        # Find or create project
        if agent.project is None:
            # Check resources
            missing = []
            for res_name, amount in config["resources"].items():
                rt = ResourceType[res_name.upper()]
                if agent.inventory.get(rt) < amount:
                    missing.append(res_name)
            if missing:
                return ActionResult(False, f"Missing resources: {missing}")

            # Consume resources
            for res_name, amount in config["resources"].items():
                rt = ResourceType[res_name.upper()]
                agent.inventory.remove(rt, amount)

            project_id = f"{agent.id}_{build_type}_{world_state.turn}"
            agent.project = Project(
                id=project_id,
                action_type=ActionType.BUILD,
                actor_id=agent.id,
                required_progress=float(config["turns"]),
                parameters={"build_type": build_type},
                target_region_id=agent.current_region,
            )

        project = agent.project
        collaborator_count = len(project.collaborators)
        progress_gain = 1.0 + collaborator_count * 0.5
        project.progress += progress_gain

        if project.progress >= project.required_progress:
            struct_id = f"struct_{project.id}"
            struct = Structure(
                id=struct_id,
                name=build_type,
                region_id=agent.current_region,
                builder_id=agent.id,
                is_complete=True,
                turns_to_build=config["turns"],
                defense_bonus=0.3 if build_type == "wall" else 0.0,
                storage_bonus=0.5 if build_type == "storehouse" else 0.0,
            )
            region.structures.append(struct)
            agent.project = None
            agent.skills["building"] = min(1.0, agent.skills.get("building", 0.3) + 0.02)
            return ActionResult(True, f"{agent.name} completed building a {build_type}!")
        else:
            return ActionResult(True, f"{agent.name} is building a {build_type} ({project.progress:.1f}/{project.required_progress:.1f}).")

    # --- TRADE ---
    def _resolve_trade(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        actor = world_state.agents.get(action.actor_id)
        target = world_state.agents.get(action.target_agent_id or "")
        if not actor or not target:
            return ActionResult(False, "Actor or target not found.")

        offer = action.parameters.get("offer", {})
        request = action.parameters.get("request", {})

        # Calculate acceptance probability
        trust = target.get_relationship(actor.id).trust
        rel_affinity = target.get_relationship(actor.id).affinity
        acceptance_prob = 0.3 + trust * 0.4 + rel_affinity * 0.2
        acceptance_prob = max(0.1, min(0.95, acceptance_prob))

        if rng.random() > acceptance_prob:
            return ActionResult(False, f"{target.name} declined the trade.")

        # Check actor has offer items
        for res_name, amount in offer.items():
            try:
                rt = ResourceType[res_name.upper()]
            except KeyError:
                continue
            if actor.inventory.get(rt) < amount:
                return ActionResult(False, f"{actor.name} lacks {res_name} for trade.")

        # Execute trade
        for res_name, amount in offer.items():
            try:
                rt = ResourceType[res_name.upper()]
            except KeyError:
                continue
            actor.inventory.remove(rt, amount)
            target.inventory.add(rt, amount)

        for res_name, amount in request.items():
            try:
                rt = ResourceType[res_name.upper()]
            except KeyError:
                continue
            target.inventory.remove(rt, amount)
            actor.inventory.add(rt, amount)

        # Update relationships
        actor.get_relationship(target.id).update_after_cooperation(0.05)
        actor.get_relationship(target.id).affinity = min(1.0, actor.get_relationship(target.id).affinity + 0.03)
        target.get_relationship(actor.id).update_after_cooperation(0.05)
        target.get_relationship(actor.id).affinity = min(1.0, target.get_relationship(actor.id).affinity + 0.03)

        return ActionResult(
            True,
            f"{actor.name} traded with {target.name}.",
            relationship_changes={target.id: {"trust": 0.05}},
        )

    # --- EXPLORE ---
    def _resolve_explore(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")

        skill = agent.skills.get("exploration", 0.3)
        curiosity = agent.personality.curiosity
        discovery_prob = 0.2 + skill * 0.3 + curiosity * 0.1

        hidden = [
            rid for rid, r in world_state.regions.items()
            if not r.is_discovered and rid not in agent.known_regions
        ]

        if rng.random() < discovery_prob and hidden:
            discovered_id = rng.choice(hidden)
            world_state.regions[discovered_id].is_discovered = True
            agent.known_regions.append(discovered_id)
            agent.add_memory(MemoryEntry(
                turn=world_state.turn,
                category="exploration",
                content=f"Discovered new region: {world_state.regions[discovered_id].name}",
                importance=0.8,
            ))
            agent.skills["exploration"] = min(1.0, agent.skills.get("exploration", 0.3) + 0.02)
            return ActionResult(True, f"{agent.name} discovered {world_state.regions[discovered_id].name}!")
        else:
            agent.add_memory(MemoryEntry(
                turn=world_state.turn,
                category="exploration",
                content="Explored but found nothing new.",
                importance=0.2,
            ))
            return ActionResult(False, f"{agent.name} explored but found nothing new.")

    # --- COMBAT ---
    def _resolve_combat(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        attacker = world_state.agents.get(action.actor_id)
        defender = world_state.agents.get(action.target_agent_id or "")
        if not attacker or not defender:
            return ActionResult(False, "Attacker or defender not found.")
        if not defender.is_alive:
            return ActionResult(False, "Target is already dead.")

        atk_skill = attacker.skills.get("combat", 0.3)
        def_skill = defender.skills.get("combat", 0.3)
        desperation = 1.0 if attacker.needs.is_desperate else 0.0

        atk_power = 1.0 * (0.5 + atk_skill) * attacker.health * (
            1 + attacker.personality.aggression + desperation * 0.3
        )

        region = world_state.regions.get(defender.current_region)
        defensibility = region.defensibility if region else 0.5
        is_defending = getattr(defender.pending_action, 'action_type', None) == ActionType.DEFEND
        def_bonus = 1.3 if is_defending else 1.0
        def_power = 1.0 * (0.5 + def_skill) * defender.health * def_bonus * (1 + defensibility)

        # Randomness
        atk_roll = atk_power * (0.8 + rng.random() * 0.4)
        def_roll = def_power * (0.8 + rng.random() * 0.4)

        damage_to_defender = max(0.0, (atk_roll - def_roll) / (def_roll + 1)) * 0.3
        damage_to_attacker = max(0.0, (def_roll - atk_roll) / (atk_roll + 1)) * 0.2

        defender.health = max(0.0, defender.health - damage_to_defender)
        attacker.health = max(0.0, attacker.health - damage_to_attacker)
        attacker.skills["combat"] = min(1.0, attacker.skills.get("combat", 0.3) + 0.01)

        # Witnesses disapprove
        witnesses = []
        if region:
            for oid in region.occupants:
                if oid not in (attacker.id, defender.id):
                    witness = world_state.agents.get(oid)
                    if witness and witness.is_alive:
                        witnesses.append(oid)
                        rel = witness.get_relationship(attacker.id)
                        rel.affinity = max(-1.0, rel.affinity - 0.15)

        success = atk_roll > def_roll
        return ActionResult(
            success,
            f"{attacker.name} attacked {defender.name}. "
            f"Defender took {damage_to_defender:.2f} damage.",
            witnesses=witnesses,
        )

    # --- HEAL ---
    def _resolve_heal(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        healer = world_state.agents.get(action.actor_id)
        patient_id = action.target_agent_id or action.actor_id
        patient = world_state.agents.get(patient_id)
        if not healer or not patient:
            return ActionResult(False, "Healer or patient not found.")

        skill = healer.skills.get("medicine", 0.3)
        healing = 0.1 + skill * 0.2

        herbs_available = healer.inventory.herbs
        if herbs_available >= 0.5:
            healer.inventory.remove(ResourceType.HERBS, 0.5)
            healing += 0.3

        patient.health = min(1.0, patient.health + healing)
        healer.skills["medicine"] = min(1.0, healer.skills.get("medicine", 0.3) + 0.01)

        if patient_id != healer.id:
            healer.get_relationship(patient_id).update_after_cooperation(0.1)
            patient.get_relationship(healer.id).update_after_cooperation(0.1)

        return ActionResult(
            True,
            f"{healer.name} healed {patient.name} for {healing:.2f} health.",
            relationship_changes={patient_id: {"trust": 0.1}} if patient_id != healer.id else {},
        )

    # --- MOVE ---
    def _resolve_move(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")

        target_region_id = action.target_region_id
        if not target_region_id:
            return ActionResult(False, "No target region specified.")

        if target_region_id not in agent.known_regions:
            return ActionResult(False, f"{agent.name} doesn't know about {target_region_id}.")

        if target_region_id not in world_state.regions:
            return ActionResult(False, f"Region {target_region_id} doesn't exist.")

        old_region = world_state.regions.get(agent.current_region)
        if old_region and agent.id in old_region.occupants:
            old_region.occupants.remove(agent.id)

        agent.current_region = target_region_id
        new_region = world_state.regions[target_region_id]
        if agent.id not in new_region.occupants:
            new_region.occupants.append(agent.id)

        # Cancel active project on move
        agent.project = None

        return ActionResult(True, f"{agent.name} moved to {new_region.name}.")

    # --- SHARE ---
    def _resolve_share(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        giver = world_state.agents.get(action.actor_id)
        receiver = world_state.agents.get(action.target_agent_id or "")
        if not giver or not receiver:
            return ActionResult(False, "Giver or receiver not found.")

        resource_name = action.parameters.get("resource", "food")
        amount = float(action.parameters.get("amount", 1.0))

        try:
            rt = ResourceType[resource_name.upper()]
        except KeyError:
            rt = ResourceType.FOOD

        actual = giver.inventory.remove(rt, amount)
        if actual <= 0:
            return ActionResult(False, f"{giver.name} has nothing to share.")

        receiver.inventory.add(rt, actual)

        giver.get_relationship(receiver.id).update_after_cooperation(0.1)
        giver.get_relationship(receiver.id).affinity = min(1.0, giver.get_relationship(receiver.id).affinity + 0.1)
        receiver.get_relationship(giver.id).update_after_cooperation(0.1)
        receiver.get_relationship(giver.id).affinity = min(1.0, receiver.get_relationship(giver.id).affinity + 0.1)
        # Track debt
        receiver.get_relationship(giver.id).debt = min(5.0, receiver.get_relationship(giver.id).debt + actual * 0.1)

        return ActionResult(
            True,
            f"{giver.name} shared {actual:.1f} {resource_name} with {receiver.name}.",
            relationship_changes={receiver.id: {"trust": 0.1, "affinity": 0.1}},
        )

    # --- REST ---
    def _resolve_rest(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")

        region = world_state.regions.get(agent.current_region)
        has_shelter = region and any(
            s.is_complete and s.name in ("shelter", "storehouse", "workshop") for s in region.structures
        )
        rest_restored = 0.3 if has_shelter else 0.2
        agent.needs.rest = max(0.0, agent.needs.rest - rest_restored)
        agent.health = min(1.0, agent.health + 0.03)

        return ActionResult(True, f"{agent.name} rested ({'with shelter' if has_shelter else 'without shelter'}).")

    # --- TEACH ---
    def _resolve_teach(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        teacher = world_state.agents.get(action.actor_id)
        student = world_state.agents.get(action.target_agent_id or "")
        if not teacher or not student:
            return ActionResult(False, "Teacher or student not found.")

        skill_name = action.parameters.get("skill", "gathering")
        teacher_skill = teacher.skills.get(skill_name, 0.0)
        student_skill = student.skills.get(skill_name, 0.0)

        if teacher_skill <= student_skill + 0.1:
            return ActionResult(False, f"{teacher.name} isn't skilled enough to teach {skill_name}.")

        gap = teacher_skill - student_skill
        openness = student.personality.openness
        learning = gap * 0.3 * (0.5 + openness * 0.5)
        new_skill = min(teacher_skill * 0.9, student_skill + learning)
        student.skills[skill_name] = new_skill

        teacher.get_relationship(student.id).update_after_cooperation(0.08)
        student.get_relationship(teacher.id).update_after_cooperation(0.1)
        student.get_relationship(teacher.id).respect = min(1.0, student.get_relationship(teacher.id).respect + 0.1)

        return ActionResult(
            True,
            f"{teacher.name} taught {student.name} {skill_name} (+{learning:.2f}).",
        )

    # --- CRAFT ---
    def _resolve_craft(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")

        craft_type = action.parameters.get("craft_type", "rope")
        skill = agent.skills.get("crafting", 0.3)

        recipes = {
            "rope": {"wood": 2.0, "skill_req": 0.0},
            "stone_tools": {"stone": 3.0, "skill_req": 0.0},
            "stone_axe": {"stone": 2.0, "wood": 1.0, "skill_req": 0.4},
            "medicine_pouch": {"herbs": 3.0, "skill_req": 0.3},
            "fishing_net": {"wood": 4.0, "skill_req": 0.5},
        }

        recipe = recipes.get(craft_type)
        if not recipe:
            return ActionResult(False, f"Unknown recipe: {craft_type}.")

        skill_req = recipe.pop("skill_req", 0.0)
        if skill < skill_req:
            recipe["skill_req"] = skill_req
            return ActionResult(False, f"{agent.name} lacks skill to craft {craft_type}.")

        # Check resources
        for res_name, amount in recipe.items():
            if res_name == "skill_req":
                continue
            try:
                rt = ResourceType[res_name.upper()]
            except KeyError:
                continue
            if agent.inventory.get(rt) < amount:
                return ActionResult(False, f"Insufficient {res_name} for crafting.")

        # Consume resources
        for res_name, amount in recipe.items():
            if res_name == "skill_req":
                continue
            try:
                rt = ResourceType[res_name.upper()]
                agent.inventory.remove(rt, amount)
            except KeyError:
                pass

        success_prob = 0.5 + skill * 0.5
        if rng.random() < success_prob:
            agent.skills["crafting"] = min(1.0, agent.skills.get("crafting", 0.3) + 0.01)
            return ActionResult(True, f"{agent.name} crafted {craft_type} successfully.")
        else:
            return ActionResult(False, f"{agent.name} failed to craft {craft_type}.")

    # --- STEAL ---
    def _resolve_steal(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        thief = world_state.agents.get(action.actor_id)
        target = world_state.agents.get(action.target_agent_id or "")
        if not thief or not target:
            return ActionResult(False, "Thief or target not found.")

        stealth = thief.skills.get("gathering", 0.3) + thief.personality.openness * 0.3
        detection = target.personality.conscientiousness * 0.5 + 0.3

        caught = rng.random() < detection / (stealth + detection)

        resource_name = action.parameters.get("resource", "food")
        try:
            rt = ResourceType[resource_name.upper()]
        except KeyError:
            rt = ResourceType.FOOD

        amount = target.inventory.get(rt) * 0.3
        if amount <= 0:
            return ActionResult(False, f"{target.name} has nothing to steal.")

        target.inventory.remove(rt, amount)
        thief.inventory.add(rt, amount)

        if caught:
            target.get_relationship(thief.id).update_after_betrayal()
            thief.get_relationship(target.id).fear = min(1.0, thief.get_relationship(target.id).fear + 0.2)
            target.emotions.append(Emotion(EmotionType.ANGER, 0.8, thief.id))
            return ActionResult(
                True,
                f"{thief.name} was caught stealing from {target.name}!",
                relationship_changes={target.id: {"trust": -0.3}},
            )

        return ActionResult(True, f"{thief.name} stole {amount:.1f} {resource_name} from {target.name}.")

    # --- SABOTAGE ---
    def _resolve_sabotage(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        saboteur = world_state.agents.get(action.actor_id)
        if not saboteur:
            return ActionResult(False, "Saboteur not found.")

        region = world_state.regions.get(action.target_region_id or saboteur.current_region)
        if not region or not region.structures:
            return ActionResult(False, "No structures to sabotage.")

        stealth = saboteur.skills.get("crafting", 0.3) + saboteur.personality.openness * 0.2
        detected = rng.random() < 0.4 - stealth * 0.2

        structure = rng.choice([s for s in region.structures if s.is_complete])
        damage = 0.2 + saboteur.skills.get("crafting", 0.3) * 0.2
        structure.durability = max(0.0, structure.durability - damage)

        builder = world_state.agents.get(structure.builder_id)
        if builder and builder.is_alive:
            builder.emotions.append(Emotion(EmotionType.ANGER, 0.7, saboteur.id))

        if detected:
            return ActionResult(
                True,
                f"{saboteur.name} was spotted sabotaging {structure.name}!",
                relationship_changes={structure.builder_id: {"trust": -0.3}},
            )
        return ActionResult(True, f"{saboteur.name} sabotaged {structure.name}.")

    # --- PERSUADE ---
    def _resolve_persuade(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        persuader = world_state.agents.get(action.actor_id)
        target = world_state.agents.get(action.target_agent_id or "")
        if not persuader or not target:
            return ActionResult(False, "Persuader or target not found.")

        skill = persuader.skills.get("persuasion", 0.3)
        extraversion = persuader.personality.extraversion
        trust = target.get_relationship(persuader.id).trust
        respect = target.get_relationship(persuader.id).respect
        persuader_power = skill + extraversion + trust + respect

        resistance = target.personality.conscientiousness + 0.5
        success = persuader_power > resistance * rng.uniform(0.8, 1.2)

        persuader.skills["persuasion"] = min(1.0, persuader.skills.get("persuasion", 0.3) + 0.01)

        if success:
            target.get_relationship(persuader.id).respect = min(1.0, trust + 0.1)
            return ActionResult(True, f"{persuader.name} successfully persuaded {target.name}.")
        return ActionResult(False, f"{target.name} resisted {persuader.name}'s persuasion.")

    # --- OBSERVE ---
    def _resolve_observe(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")

        region = world_state.regions.get(agent.current_region)
        if not region:
            return ActionResult(False, "Region not found.")

        parts = [f"Region: {region.name}"]
        for rt, node in region.resources.items():
            parts.append(f"  {rt.value}: {node.amount:.1f}/{node.max_amount:.1f}")
        parts.append(f"Occupants: {', '.join(region.occupants)}")
        parts.append(f"Structures: {len(region.structures)}")

        return ActionResult(True, " | ".join(parts))

    # --- DEFEND ---
    def _resolve_defend(self, action: Action, world_state: "WorldState", rng: Any) -> ActionResult:
        agent = world_state.agents.get(action.actor_id)
        if not agent:
            return ActionResult(False, "Agent not found.")
        agent.pending_action = action
        return ActionResult(True, f"{agent.name} is defending.")
