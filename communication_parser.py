"""
communication_parser.py — Parses and processes agent communication.

Includes EmotionalSpeechModifier which adds emotional context to the
communication pipeline and detects emotional authenticity (masking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent_state import Agent


# ---------------------------------------------------------------------------
# Communication message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A message sent from one agent to another (or broadcast)."""
    sender_id: str
    content: str
    recipients: List[str] = field(default_factory=list)  # empty = broadcast
    turn: int = 0

    # Filled in by EmotionalSpeechModifier
    detected_tone: Optional[str] = None          # Emotion name detected in the text
    actual_emotion: Optional[str] = None         # Agent's true dominant emotion
    is_authentic: Optional[bool] = None          # Does tone match actual emotion?
    is_masking: Optional[bool] = None            # Agent hiding their real emotion?


# ---------------------------------------------------------------------------
# Communication result
# ---------------------------------------------------------------------------

@dataclass
class CommunicationResult:
    """Holds the outcome of processing a single message."""
    message: Message
    recipients_notified: List[str] = field(default_factory=list)
    masking_detected_by: List[str] = field(default_factory=list)  # agent IDs who noticed


# ---------------------------------------------------------------------------
# Emotional speech modifier
# ---------------------------------------------------------------------------

# Simple keyword-to-emotion mapping for tone detection
_TONE_KEYWORDS: Dict[str, List[str]] = {
    "fear": ["afraid", "scared", "worried", "terrified", "nervous", "danger", "threat"],
    "anger": ["furious", "angry", "enraged", "outraged", "hate", "betrayed", "attack"],
    "hope": ["hope", "believe", "optimistic", "future", "together", "better", "trust"],
    "grief": ["sad", "lost", "mourn", "miss", "alone", "grief", "sorry"],
    "gratitude": ["thank", "grateful", "appreciate", "glad", "fortunate"],
    "pride": ["proud", "achieved", "accomplished", "strong", "superior", "best"],
    "envy": ["jealous", "unfair", "deserve", "why do they", "they have"],
    "loneliness": ["alone", "isolated", "no one", "abandoned", "forgotten"],
    "disgust": ["disgusting", "repulsive", "vile", "horrible", "awful"],
}


class EmotionalSpeechModifier:
    """Analyzes and modifies agent speech based on emotional state."""

    def process(
        self,
        message: Message,
        sender: Agent,
        recipients: List[Agent],
    ) -> CommunicationResult:
        """Process a message through the emotional speech pipeline.

        1. Detect the tone of the message content.
        2. Compare to the sender's actual dominant emotion.
        3. Determine if the sender is masking emotions.
        4. Check if perceptive recipients notice the masking.
        5. Return a CommunicationResult.
        """
        # Detect tone from message content
        detected_tone = self._detect_tone(message.content)
        message.detected_tone = detected_tone

        # Determine the sender's actual dominant emotion
        dom = sender.dominant_emotion()
        actual_emotion = dom.emotion_type.value if dom else None
        message.actual_emotion = actual_emotion

        # Authenticity check
        if detected_tone is not None and actual_emotion is not None:
            message.is_authentic = (detected_tone == actual_emotion)
        else:
            message.is_authentic = True  # No strong signals → assume authentic

        # Masking: hidden emotion only if honesty < 0.5 AND tone doesn't match
        masking = (
            message.is_authentic is False
            and sender.personality.honesty < 0.5
        )
        message.is_masking = masking

        # Which recipients detect the masking?
        masking_detectors: List[str] = []
        if masking:
            for recipient in recipients:
                if self._can_detect_masking(recipient, sender):
                    masking_detectors.append(recipient.agent_id)
                    recipient.add_memory(
                        content=(
                            f"I noticed {sender.name} seemed to be hiding something — "
                            f"their words didn't match their demeanor."
                        ),
                        importance=0.6,
                        turn=message.turn,
                    )

        result = CommunicationResult(
            message=message,
            recipients_notified=[r.agent_id for r in recipients],
            masking_detected_by=masking_detectors,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_tone(self, content: str) -> Optional[str]:
        """Detect the dominant emotional tone in a message by keyword matching."""
        lower = content.lower()
        scores: Dict[str, int] = {}
        for emotion, keywords in _TONE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in lower)
            if count:
                scores[emotion] = count
        if not scores:
            return None
        return max(scores, key=lambda e: scores[e])

    def _can_detect_masking(self, recipient: Agent, sender: Agent) -> bool:
        """Return True if the recipient is perceptive enough to notice masking."""
        # High conscientiousness → perceptive
        if recipient.personality.conscientiousness > 0.65:
            return True
        # Many interactions → familiar enough to notice
        rel = recipient.relationships.get(sender.agent_id)
        if rel and rel.interaction_count >= 5:
            return True
        return False

    def build_emotional_prompt_context(self, agent: Agent) -> str:
        """Build a prompt fragment describing the agent's emotional speech context."""
        lines: List[str] = []
        lines.append(agent.emotion_summary())
        lines.append(agent.speech_style_prompt())
        complex_summary = agent.complex_emotion_summary()
        if complex_summary:
            lines.append(complex_summary)
        mood_desc = agent.mood.prompt_description()
        lines.append(mood_desc)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simple communication pipeline
# ---------------------------------------------------------------------------

class CommunicationPipeline:
    """Processes all agent messages for a single turn."""

    def __init__(self) -> None:
        self.modifier = EmotionalSpeechModifier()

    def process_turn_messages(
        self,
        messages: List[Message],
        agents_by_id: Dict[str, Agent],
    ) -> List[CommunicationResult]:
        """Process all messages and return results."""
        results: List[CommunicationResult] = []
        for msg in messages:
            sender = agents_by_id.get(msg.sender_id)
            if sender is None:
                continue
            # Treat an empty recipient list as a broadcast to all agents except the sender
            if msg.recipients:
                recipients = [
                    agents_by_id[rid]
                    for rid in msg.recipients
                    if rid in agents_by_id
                ]
            else:
                recipients = [
                    agent
                    for aid, agent in agents_by_id.items()
                    if aid != msg.sender_id
                ]
            result = self.modifier.process(msg, sender, recipients)
            results.append(result)
            # Update interaction counts
            for recipient in recipients:
                rel = sender.get_or_create_relationship(recipient.agent_id)
                rel.interaction_count += 1
                rel2 = recipient.get_or_create_relationship(sender.agent_id)
                rel2.interaction_count += 1
        return results
