from __future__ import annotations

import re
from typing import Any


class SpeechStyleRealizer:
    """Applies surface-language operators to baseline semantic content."""

    _chinese_prefixes = {
        "plain": "我就实在说吧，",
        "colloquial": "说白了，",
        "formal": "谨慎地说，",
        "academic": "从证据和逻辑上说，",
        "philosophical": "往深处说，",
        "technical": "先把问题拆开：",
        "aristocratic": "容我直言，",
        "confrontational": "别绕弯子。",
    }
    _english_prefixes = {
        "plain": "Plainly, ",
        "colloquial": "Look, ",
        "formal": "To state this carefully, ",
        "academic": "On the available evidence, ",
        "philosophical": "At a deeper level, ",
        "technical": "Break the problem into parts: ",
        "aristocratic": "Permit me to be direct: ",
        "confrontational": "Do not dodge the point. ",
    }

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def realize(
        self,
        semantic_text: str,
        style: dict[str, Any],
        *,
        mature_fiction: bool,
    ) -> tuple[str, dict[str, Any]]:
        text = " ".join(semantic_text.split())[:1_600]
        if not text:
            return "", {"applied": False, "reason": "empty_semantic_text"}
        register = str(style.get("voice_register", "neutral"))
        directness = float(style.get("directness", 0.5))
        roughness = float(style.get("roughness", 0.1))
        warmth = float(style.get("warmth", 0.5))
        abstraction = float(style.get("philosophical_abstraction", 0.2))
        jargon = float(style.get("technical_jargon", 0.1))
        verbosity = float(style.get("verbosity", 0.5))
        chinese = self._is_chinese(text)
        if float(style.get("sentence_complexity", 0.5)) <= 0.32:
            text = text.replace("；", "。").replace(";", ".")
        if verbosity <= 0.3:
            separators = r"(?<=[。！？.!?])"
            sentences = [part for part in re.split(separators, text) if part.strip()]
            text = "".join(sentences[:2]).strip()
        prefix = (
            self._chinese_prefixes.get(register, "")
            if chinese
            else self._english_prefixes.get(register, "")
        )
        if directness >= 0.78 and register == "neutral":
            prefix = "直说吧，" if chinese else "Directly: "
        if roughness >= 0.72:
            rough_prefix = (
                "少拿这套屁话压人。" if mature_fiction else "少来这套。"
            ) if chinese else (
                "Cut the damn posturing. " if mature_fiction else "Drop the posturing. "
            )
            prefix = f"{rough_prefix}{prefix}"
        suffix = ""
        if abstraction >= 0.72 and register != "philosophical":
            suffix += " 归根结底，这是选择与代价的关系。" if chinese else (
                " Ultimately, this is about choice and consequence."
            )
        if jargon >= 0.72 and register != "technical":
            suffix += " 先分清输入、约束和结果。" if chinese else (
                " Separate the inputs, constraints, and outcomes."
            )
        if warmth >= 0.72:
            suffix += " 我愿意把话说明白。" if chinese else " I am willing to explain it clearly."
        rendered = f"{prefix}{text}{suffix}"[:1_600]
        return rendered, {
            "applied": bool(prefix or suffix or verbosity <= 0.3),
            "voice_register": register,
            "mature_roughness_enabled": mature_fiction and roughness >= 0.72,
            "verbal_habits_are_constraints_not_inserted_phrases": True,
        }
