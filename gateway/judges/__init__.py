"""LLM judges shared by the semantic controls (SPEC "Control catalog" → "Judges")."""

from gateway.judges.client import JudgeClient, JudgeUnavailableError

__all__ = ["JudgeClient", "JudgeUnavailableError"]
