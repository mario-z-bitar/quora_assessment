"""Command-line pipeline for analyzing Poe AI conversations.

This script ingests raw conversation JSON, computes deterministic
conversation metrics, classifies conversation types, calls an LLM to
label topics, and writes summarized outputs for downstream dashboards.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import fastapi_poe as fp
except ImportError:  # pragma: no cover - optional dependency
    fp = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    """Lightweight representation for a single message."""

    role: str
    content: str
    timestamp: Optional[int]


@dataclass(frozen=True)
class Conversation:
    """Container for a full conversation history."""

    conversation_id: str
    messages: List[Message]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Conversation":
        messages = [
            Message(
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
            )
            for msg in payload.get("messages", [])
        ]
        return cls(
            conversation_id=payload.get("conversation_id", ""),
            messages=messages,
        )


@dataclass(frozen=True)
class ConversationMetrics:
    """Deterministic measurements derived from message contents."""

    conversation_id: str
    message_count: int
    user_message_count: int
    assistant_message_count: int
    total_character_count: int
    total_word_count: int
    unique_roles: int
    duration_seconds: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_count": self.message_count,
            "user_message_count": self.user_message_count,
            "assistant_message_count": self.assistant_message_count,
            "total_character_count": self.total_character_count,
            "total_word_count": self.total_word_count,
            "unique_roles": self.unique_roles,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class TopicAnnotation:
    """LLM-derived categorisation for a conversation."""

    topic: str
    intent: str
    confidence: float
    rationale: str
    raw_response: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "intent": self.intent,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
        }


class KeywordTopicHeuristic:
    """Fallback keyword matching so the pipeline still runs without an LLM."""

    KEYWORDS = {
        "customer_support": ["refund", "order", "support", "issue", "problem"],
        "technical_help": ["bug", "error", "code", "debug", "crash"],
        "creative_writing": ["story", "poem", "write", "novel", "plot"],
        "education_research": ["explain", "homework", "study", "research", "paper"],
        "planning_productivity": ["plan", "schedule", "organize", "productivity", "goal"],
        "entertainment": ["movie", "game", "music", "song", "celebrity"],
        "personal_advice": ["relationship", "career", "decision", "advice", "life"],
    }

    GENERATIVE_HINTS = [
        "write",
        "story",
        "poem",
        "draft",
        "compose",
        "create",
        "generate",
        "design",
        "brainstorm",
        "outline",
        "fiction",
    ]

    def label(self, conversation: Conversation) -> TopicAnnotation:
        content = " ".join(msg.content.lower() for msg in conversation.messages)
        best_topic: Optional[str] = None
        best_score = 0
        for topic, keywords in self.KEYWORDS.items():
            score = sum(content.count(keyword) for keyword in keywords)
            if score > best_score:
                best_topic = topic
                best_score = score
        if not best_topic or best_score == 0:
            best_topic = "general_qna"
        rationale = "keyword heuristic" if best_score else "default fallback"
        confidence = 0.35 if best_score else 0.1
        intent = self._infer_intent(content)
        return TopicAnnotation(topic=best_topic, intent=intent, confidence=confidence, rationale=rationale, raw_response=None)

    def _infer_intent(self, content: str) -> str:
        if any(keyword in content for keyword in self.GENERATIVE_HINTS):
            return "generative"
        return "fact_finding"


class LLMTopicLabeler:
    """Adapter around Poe's sync API with caching and fallbacks."""

    def __init__(
        self,
        enabled: bool = True,
        bot_name: str = "Claude-Sonnet-4.5",
        api_key: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.enabled = enabled and fp is not None
        self.bot_name = bot_name
        self.api_key = api_key or os.getenv("POE_API_KEY")
        self.cache_path = cache_path
        self.cache: Dict[str, TopicAnnotation] = {}
        if cache_path:
            self._load_cache(cache_path)
        if self.enabled and not self.api_key:
            LOGGER.warning("POE_API_KEY is not set; LLM topic labelling disabled")
            self.enabled = False
        if not self.enabled:
            LOGGER.info("Using keyword heuristic for topic labelling")
            self.heuristic = KeywordTopicHeuristic()

    def _hash_conversation(self, conversation: Conversation) -> str:
        digest = hashlib.sha256()
        digest.update(conversation.conversation_id.encode("utf-8"))
        for msg in conversation.messages:
            digest.update(msg.role.encode("utf-8"))
            digest.update(msg.content.encode("utf-8"))
        return digest.hexdigest()

    def _load_cache(self, path: Path) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw_cache = json.load(handle)
        except json.JSONDecodeError:
            LOGGER.warning("Topic cache at %s is corrupted; starting fresh", path)
            raw_cache = {}
        for key, payload in raw_cache.items():
            self.cache[key] = TopicAnnotation(
                topic=payload.get("topic", "unknown"),
                intent=payload.get("intent", "fact_finding"),
                confidence=float(payload.get("confidence", 0.0)),
                rationale=payload.get("rationale", ""),
                raw_response=payload.get("raw_response"),
            )

    def _write_cache(self) -> None:
        if not self.cache_path:
            return
        serialized = {key: value.to_dict() for key, value in self.cache.items()}
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(serialized, handle, indent=2)

    def label(self, conversation: Conversation) -> TopicAnnotation:
        cache_key = self._hash_conversation(conversation)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not self.enabled:
            annotation = self.heuristic.label(conversation)
            self.cache[cache_key] = annotation
            self._write_cache()
            return annotation
        prompt = self._build_prompt(conversation)
        try:
            response_text = self._call_poe(prompt)
            annotation = self._parse_response(response_text)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("LLM labelling failed: %s", exc)
            annotation = TopicAnnotation(
                topic="llm_error",
                intent="fact_finding",
                confidence=0.0,
                rationale=f"fallback after error: {exc}",
                raw_response=None,
            )
        self.cache[cache_key] = annotation
        self._write_cache()
        return annotation

    def _build_prompt(self, conversation: Conversation) -> str:
        formatted_turns = []
        for index, msg in enumerate(conversation.messages, start=1):
            prefix = "User" if msg.role == "user" else "Assistant"
            formatted_turns.append(f"Turn {index} ({prefix}): {msg.content.strip()}")
        conversation_block = "\n".join(formatted_turns)
        instructions = (
            "You are an analyst summarizing Poe chat conversations. "
            "Return a compact JSON object with keys topic (one to three words), "
            "intent (fact_finding or generative), confidence (0-1 float), and "
            "rationale (short phrase)."
        )
        return f"{instructions}\n\nConversation:\n{conversation_block}\n\nJSON:"

    def _call_poe(self, prompt: str) -> str:
        assert fp is not None  # nosec - guarded in __init__
        if not self.api_key:
            raise RuntimeError("POE_API_KEY is not configured")
        messages = [
            fp.ProtocolMessage(role="system", content="You label chat conversations succinctly."),
            fp.ProtocolMessage(role="user", content=prompt),
        ]
        response_chunks: List[str] = []
        for partial in fp.get_bot_response_sync(
            messages=messages,
            bot_name=self.bot_name,
            api_key=self.api_key,
        ):
            text = getattr(partial, "text", None)
            if text:
                response_chunks.append(text)
                continue
            content = getattr(partial, "content", None)
            if isinstance(content, str) and content:
                response_chunks.append(content)
                continue
            if isinstance(partial, str):
                response_chunks.append(partial)
        return "".join(response_chunks).strip()

    def _parse_response(self, response_text: str) -> TopicAnnotation:
        normalized = self._strip_markdown_code_fence(response_text)
        payload = self._coerce_json(normalized)
        topic = payload.get("topic", "unknown").strip().lower().replace(" ", "_")
        intent = payload.get("intent", "fact_finding").strip().lower().replace(" ", "_")
        if intent not in {"fact_finding", "generative"}:
            intent = "fact_finding"
        confidence = float(payload.get("confidence", 0.3))
        rationale = payload.get("rationale", "no rationale provided").strip()
        return TopicAnnotation(topic=topic, intent=intent, confidence=confidence, rationale=rationale, raw_response=response_text)

    def _strip_markdown_code_fence(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                # drop opening fence (with optional language tag) and closing fence if present
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                stripped = "\n".join(lines).strip()
        return stripped

    def _coerce_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            headline = text.split("\n", 1)[0][:60].strip()
            return {
                "topic": headline or "unknown",
                "intent": "fact_finding",
                "confidence": 0.3,
                "rationale": "non-json llm response",
            }


def ingest_conversations(raw: Dict[str, Any]) -> List[Conversation]:
    conversations = raw.get("conversations", [])
    return [Conversation.from_dict(item) for item in conversations]


def compute_metrics(conversation: Conversation) -> ConversationMetrics:
    messages = conversation.messages
    message_count = len(messages)
    user_message_count = sum(1 for msg in messages if msg.role == "user")
    assistant_message_count = sum(1 for msg in messages if msg.role == "assistant")
    total_character_count = sum(len(msg.content) for msg in messages)
    total_word_count = sum(len(msg.content.split()) for msg in messages)
    unique_roles = len({msg.role for msg in messages})
    timestamps = [msg.timestamp for msg in messages if isinstance(msg.timestamp, int)]
    duration_seconds = None
    if timestamps:
        duration_seconds = max(timestamps) - min(timestamps)
        if duration_seconds < 0:
            duration_seconds = None
    return ConversationMetrics(
        conversation_id=conversation.conversation_id,
        message_count=message_count,
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        total_character_count=total_character_count,
        total_word_count=total_word_count,
        unique_roles=unique_roles,
        duration_seconds=duration_seconds,
    )


def classify_conversation_type(metrics: ConversationMetrics) -> str:
    """Label interaction style based on deterministic metrics."""

    if metrics.message_count <= 3 or metrics.total_character_count < 600:
        return "quick_qna"
    if metrics.message_count <= 10 or metrics.total_character_count < 2000:
        return "multi_turn_help"
    return "deep_dive_dialogue"


def aggregate_insights(
    metrics_list: Iterable[ConversationMetrics],
    topic_annotations: Iterable[TopicAnnotation],
    conversation_types: Iterable[str],
) -> Dict[str, Any]:
    metrics_list = list(metrics_list)
    topic_annotations = list(topic_annotations)
    conversation_types = list(conversation_types)
    total = len(metrics_list)
    if total == 0:
        return {
            "generated_at": dt.datetime.utcnow().isoformat(),
            "total_conversations": 0,
            "notes": "No conversations processed",
        }
    avg_messages = sum(m.message_count for m in metrics_list) / total
    avg_characters = sum(m.total_character_count for m in metrics_list) / total
    median_duration = None
    durations = [m.duration_seconds for m in metrics_list if m.duration_seconds is not None]
    if durations:
        durations.sort()
        mid = len(durations) // 2
        if len(durations) % 2 == 0:
            median_duration = (durations[mid - 1] + durations[mid]) / 2
        else:
            median_duration = durations[mid]
    type_counts = Counter(conversation_types)
    topic_counts = Counter(annotation.topic for annotation in topic_annotations)
    intent_counts = Counter(annotation.intent for annotation in topic_annotations)
    actionable = []
    if type_counts.get("deep_dive_dialogue", 0) / total > 0.3:
        actionable.append("Consider expanding context window for long-form dialogues")
    if type_counts.get("quick_qna", 0) / total > 0.5:
        actionable.append("Optimize quick-answer flows and canned responses")
    dominant_topic, dominant_count = (topic_counts.most_common(1)[0] if topic_counts else (None, 0))
    if dominant_topic and dominant_count / total > 0.2:
        actionable.append(f"Prioritise fine-tuning on {dominant_topic} content")
    if intent_counts.get("generative", 0) / total > 0.4:
        actionable.append("Invest in creative-assistant tooling for generative chats")
    elif intent_counts.get("fact_finding", 0) / total > 0.7:
        actionable.append("Improve knowledge retrieval and factual grounding")
    return {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "total_conversations": total,
        "average_messages": avg_messages,
        "average_character_count": avg_characters,
        "median_duration_seconds": median_duration,
        "conversation_type_distribution": type_counts,
        "top_topics": topic_counts.most_common(10),
        "intent_distribution": intent_counts,
        "action_items": actionable,
    }


def write_output(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Poe conversations into actionable insights")
    parser.add_argument("--input", required=True, help="Path to the raw conversations JSON file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where processed JSON outputs will be written",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional JSON cache file for LLM topic annotations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of conversations processed (e.g. 60)",
    )
    parser.add_argument(
        "--poe-bot-name",
        default="Claude-Sonnet-4.5",
        help="Poe bot name used for topic labelling",
    )
    parser.add_argument(
        "--poe-api-key",
        default=None,
        help="Override Poe API key (falls back to POE_API_KEY env var)",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Use keyword heuristics instead of calling an LLM",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache_path) if args.cache_path else None

    LOGGER.info("Loading conversations from %s", input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    conversations = ingest_conversations(raw_payload)
    if args.limit is not None:
        conversations = conversations[: args.limit]
    LOGGER.info("Loaded %s conversations", len(conversations))

    labeler = LLMTopicLabeler(
        enabled=not args.disable_llm,
        bot_name=args.poe_bot_name,
        api_key=args.poe_api_key,
        cache_path=cache_path,
    )

    per_conversation_output: List[Dict[str, Any]] = []
    metrics_entries: List[ConversationMetrics] = []
    annotations: List[TopicAnnotation] = []
    conversation_types: List[str] = []

    for conversation in conversations:
        metrics = compute_metrics(conversation)
        annotation = labeler.label(conversation)
        convo_type = classify_conversation_type(metrics)

        metrics_entries.append(metrics)
        annotations.append(annotation)
        conversation_types.append(convo_type)

        per_conversation_output.append(
            {
                **metrics.to_dict(),
                "topic": annotation.topic,
                "conversation_intent": annotation.intent,
                "topic_confidence": annotation.confidence,
                "topic_rationale": annotation.rationale,
                "conversation_type": convo_type,
            }
        )

    aggregate = aggregate_insights(metrics_entries, annotations, conversation_types)
    per_conversation_path = output_dir / "per_conversation_metrics.json"
    aggregate_path = output_dir / "aggregate_summary.json"

    LOGGER.info("Writing per-conversation metrics to %s", per_conversation_path)
    write_output(per_conversation_path, {"conversations": per_conversation_output})
    LOGGER.info("Writing aggregate summary to %s", aggregate_path)
    write_output(aggregate_path, aggregate)

    LOGGER.info("Processing complete")


if __name__ == "__main__":
    main()
