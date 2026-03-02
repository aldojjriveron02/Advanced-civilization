from __future__ import annotations
import re
import json
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    temperature: float = 0.8
    max_tokens: int = 1500


class LLMInterface:
    SYSTEM_PROMPT = (
        "You are an autonomous agent in a survival civilization simulation. "
        "Respond in this exact format:\n"
        "<private_reasoning>\n"
        "Your internal thoughts and reasoning here\n"
        "</private_reasoning>\n"
        "<actions>\n"
        '[{"action_type": "GATHER", "target_resource": "FOOD", "parameters": {}}]\n'
        "</actions>\n"
        "<speech>\n"
        "What you say aloud (empty if silent)\n"
        "</speech>"
    )

    def __init__(self, config: LLMConfig = None) -> None:
        self.config = config or LLMConfig()
        self.call_count: int = 0
        self.total_tokens: int = 0
        self._client = None
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except Exception:
            self._client = None

    def call(self, prompt: str) -> str:
        if self._client is None:
            return ""
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            self.call_count += 1
            usage = getattr(response, "usage", None)
            if usage:
                self.total_tokens += getattr(usage, "total_tokens", 0)
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    @staticmethod
    def parse_response(response: str) -> dict:
        reasoning = ""
        actions = []
        speech = ""

        reasoning_match = re.search(
            r"<private_reasoning>(.*?)</private_reasoning>", response, re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        actions_match = re.search(r"<actions>(.*?)</actions>", response, re.DOTALL)
        if actions_match:
            raw = actions_match.group(1).strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    actions = parsed
                elif isinstance(parsed, dict):
                    actions = [parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        speech_match = re.search(r"<speech>(.*?)</speech>", response, re.DOTALL)
        if speech_match:
            speech = speech_match.group(1).strip()

        return {"reasoning": reasoning, "actions": actions, "speech": speech}
