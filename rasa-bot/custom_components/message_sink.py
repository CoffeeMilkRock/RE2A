# rasa-bot/custom_components/message_sink.py
from __future__ import annotations
from typing import Any, Dict, List, Text
import os, requests

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.nlu.training_data.message import Message

@DefaultV1Recipe.register(
    component_types=[GraphComponent],   # tell the DefaultV1 recipe what this is
    is_trainable=False,                 # we don't train this component
)
class MessageSink(GraphComponent):
    """Send each incoming user message to your API for embedding/logging."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "api_base": os.getenv("REA_API_BASE", "http://localhost:8008/api/v2"),
            "timeout": 5,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        self.api_base = config.get("api_base") or "http://localhost:8008/api/v2"
        self.timeout = int(config.get("timeout", 5))

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        execution_context: ExecutionContext,  # required by interface
    ) -> "MessageSink":
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        for m in messages:
            text = (m.get("text") or "").strip()
            if not text:
                continue

            payload = {
                "conversation_id": m.get("conversation_id") or "unknown",
                "user_id": m.get("sender_id") or "unknown",
                "role": "user",
                "text": text,
                "intent": (m.get("intent") or {}).get("name"),
                "entities": m.get("entities") or [],
                "slots": {},  # slots not available at NLU stage
                "extra": {"input_channel": m.get("input_channel")},
            }

            try:
                requests.post(
                    f"{self.api_base}/conversation/message",
                    json=payload,
                    timeout=self.timeout,
                )
            except Exception:
                # never break the pipeline on logging errors
                pass

        return messages
