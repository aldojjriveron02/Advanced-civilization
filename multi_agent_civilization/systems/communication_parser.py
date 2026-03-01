from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models.agent_state import Emotion, EmotionType, MemoryEntry
from models.world_state import WeatherType

if TYPE_CHECKING:
    from models.agent_state import Agent
    from models.world_state import WorldState


class CommunicationParser:
    # Intent patterns
    INTENT_PATTERNS = {
        "trade_offer": re.compile(r"\b(offer|trade|exchange|give.{0,10}for)\b", re.IGNORECASE),
        "request_help": re.compile(r"\b(help|assist|need|please)\b", re.IGNORECASE),
        "threat": re.compile(r"\b(threaten|attack|hurt|die|kill)\b", re.IGNORECASE),
        "alliance_proposal": re.compile(r"\b(ally|alliance|together|join)\b", re.IGNORECASE),
        "information_sharing": re.compile(r"\b(found|discovered|there is|located)\b", re.IGNORECASE),
        "accusation": re.compile(r"\b(lied|betrayed|stole|cheated)\b", re.IGNORECASE),
        "greeting": re.compile(r"\b(hello|hi|greet|meet)\b", re.IGNORECASE),
        "farewell": re.compile(r"\b(goodbye|farewell|leaving)\b", re.IGNORECASE),
        "boast": re.compile(r"\b(I am|strongest|best|I have)\b", re.IGNORECASE),
        "mourn": re.compile(r"\b(dead|died|loss|grief|sad)\b", re.IGNORECASE),
        "command": re.compile(r"\b(go|do|must|shall)\b", re.IGNORECASE),
        "gossip": re.compile(r"\b(heard|apparently|rumor|they said)\b", re.IGNORECASE),
    }

    TONE_PATTERNS = {
        "hostile": re.compile(r"\b(attack|kill|hurt|destroy|crush)\b", re.IGNORECASE),
        "friendly": re.compile(r"\b(thank|help|friend|appreciate|grateful)\b", re.IGNORECASE),
        "urgent": re.compile(r"\b(hurry|quick|danger|emergency|now)\b", re.IGNORECASE),
        "deceptive": re.compile(r"\b(trust me|I promise)\b", re.IGNORECASE),
        "grieving": re.compile(r"\b(sad|died|loss|grief|mourn)\b", re.IGNORECASE),
        "confident": re.compile(r"\b(I will|I am|certainly|definitely)\b", re.IGNORECASE),
        "fearful": re.compile(r"\b(afraid|fear|scared|terrified)\b", re.IGNORECASE),
    }

    def parse_message(
        self,
        speaker: "Agent",
        speech: str,
        world_state: "WorldState",
        current_turn: int,
        rng: Any,
    ) -> List[Dict]:
        if not speech.strip():
            return []

        # Detect whisper
        whisper_match = re.search(r"\[to ([A-Za-z]+)\]", speech)
        whisper_target_name = whisper_match.group(1) if whisper_match else None
        whisper_target_id = None
        if whisper_target_name:
            for aid, agent in world_state.agents.items():
                if agent.name.lower() == whisper_target_name.lower():
                    whisper_target_id = aid
                    break

        # Classify intent
        intent = "general"
        for intent_name, pattern in self.INTENT_PATTERNS.items():
            if pattern.search(speech):
                intent = intent_name
                break

        # Detect tone
        tone = "neutral"
        for tone_name, pattern in self.TONE_PATTERNS.items():
            if pattern.search(speech):
                tone = tone_name
                break

        # Get region occupants as potential recipients
        region = world_state.regions.get(speaker.current_region)
        region_agents = []
        if region:
            region_agents = [
                world_state.agents[oid]
                for oid in region.occupants
                if oid != speaker.id and world_state.agents.get(oid) and world_state.agents[oid].is_alive
            ]

        # Determine recipients
        if whisper_target_id:
            target_agent = world_state.agents.get(whisper_target_id)
            recipients = [target_agent] if target_agent else []
            # 15% chance others overhear
            for other in region_agents:
                if other.id != whisper_target_id and rng.random() < 0.15:
                    recipients.append(other)
        else:
            recipients = region_agents

        results = []
        for recipient in recipients:
            if not recipient or not recipient.is_alive:
                continue

            trust = recipient.get_relationship(speaker.id).trust

            # Misinterpretation
            misinterpret_prob = 0.08
            misinterpret_prob -= trust * 0.05
            if recipient.needs.is_desperate:
                misinterpret_prob += 0.05
            dom = recipient.dominant_emotion
            if dom and dom.emotion_type in (EmotionType.FEAR, EmotionType.ANGER):
                misinterpret_prob += 0.05
            if world_state.weather in (WeatherType.STORM, WeatherType.FOG):
                misinterpret_prob += 0.03
            misinterpret_prob = max(0.0, min(0.5, misinterpret_prob))
            is_misinterpreted = rng.random() < misinterpret_prob

            # Apply tone effects
            self._apply_tone_emotion(recipient, speaker, tone, trust, rng)

            # Process claims
            claims = self._extract_claims(speech, speaker, recipient, world_state, trust, rng)

            # Add memory
            importance = 0.4 + (0.2 if intent in ("threat", "alliance_proposal", "accusation") else 0.0)
            recipient.add_memory(MemoryEntry(
                turn=current_turn,
                category="communication",
                content=f"{speaker.name} said: '{speech[:80]}'" + (" [MISHEARD]" if is_misinterpreted else ""),
                importance=importance,
                related_agents=[speaker.id],
                emotional_valence=0.3 if tone == "friendly" else (-0.3 if tone in ("hostile", "threat") else 0.0),
            ))

            results.append({
                "speaker_id": speaker.id,
                "recipient_id": recipient.id,
                "intent": intent,
                "tone": tone,
                "content": speech,
                "is_misinterpreted": is_misinterpreted,
                "claims": claims,
            })

        return results

    def _apply_tone_emotion(self, recipient: "Agent", speaker: "Agent", tone: str, trust: float, rng: Any) -> None:
        if tone == "hostile":
            if recipient.personality.aggression > 0.5:
                recipient.emotions.append(Emotion(EmotionType.ANGER, 0.6, speaker.id))
            else:
                recipient.emotions.append(Emotion(EmotionType.FEAR, 0.5, speaker.id))
        elif tone == "friendly" and trust > 0.5:
            recipient.emotions.append(Emotion(EmotionType.GRATITUDE, 0.4, speaker.id))
        elif tone == "deceptive":
            if trust < 0.4:
                # Low trust speaker saying "trust me" reduces trust
                rel = recipient.get_relationship(speaker.id)
                rel.trust = max(0.0, rel.trust - 0.05)
        elif tone == "grieving":
            recipient.emotions.append(Emotion(EmotionType.GRIEF, 0.3, speaker.id))

    def _is_gossip_worthy(self, intent: str, content: str, speaker: "Agent") -> bool:
        if intent in ("information_sharing", "accusation", "gossip"):
            if speaker.personality.extraversion > 0.5:
                return True
        return False

    def _extract_claims(
        self, speech: str, speaker: "Agent", recipient: "Agent",
        world_state: "WorldState", trust: float, rng: Any
    ) -> List[Dict]:
        claims = []
        # Simple resource/region mentions
        resource_pattern = re.compile(r"\b(food|water|wood|stone|herbs|metal)\b", re.IGNORECASE)
        region_pattern = re.compile(r"\b(forest|river|hills|coast|cave)\b", re.IGNORECASE)

        for match in resource_pattern.finditer(speech):
            resource = match.group(1).lower()
            accepted = rng.random() < (0.3 + trust * 0.5)
            claims.append({"type": "resource_mention", "resource": resource, "accepted": accepted})

        for match in region_pattern.finditer(speech):
            region_name = match.group(1).lower()
            accepted = rng.random() < (0.3 + trust * 0.5)
            claims.append({"type": "region_mention", "region": region_name, "accepted": accepted})

        return claims

    def _process_strategic_silence(self, world_state: "WorldState", speakers_this_turn: List[str]) -> None:
        for aid, agent in world_state.agents.items():
            if not agent.is_alive or aid in speakers_this_turn:
                continue
            for other_id, rel in agent.relationships.items():
                other = world_state.agents.get(other_id)
                if not other or not other.is_alive:
                    continue
                # If expected to hear from this agent but didn't
                interaction_count = len([
                    m for m in agent.memories
                    if other_id in m.related_agents and m.turn >= world_state.turn - 3
                ])
                if interaction_count > 2 or rel.fear > 0.5 or rel.affinity > 0.7:
                    if other_id not in speakers_this_turn:
                        agent.add_memory(MemoryEntry(
                            turn=world_state.turn,
                            category="social",
                            content=f"{other.name} was notably silent today.",
                            importance=0.3,
                            related_agents=[other_id],
                        ))
