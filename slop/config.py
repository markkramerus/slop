"""
config.py — API client configuration for any OpenAI-compatible endpoint.

Set via environment variables or pass a Config object explicitly.

Environment variables:
  SLOP_API_BASE_URL   Base URL for the chat completions API
                      Default: https://api.openai.com/v1
  SLOP_API_KEY        API key
  SLOP_CHAT_MODEL     Model name for chat/generation
                      Default: gpt-4o
  SLOP_EMBED_MODEL    Model name for embeddings
                      Default: text-embedding-3-small
  SLOP_MAX_TOKENS     Maximum tokens for generated comment
                      Default: 1024
  SLOP_TEMPERATURE    Default sampling temperature (overridden per vector)
                      Default: 0.9
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    api_base_url: str = field(
        default_factory=lambda: os.getenv("SLOP_API_BASE_URL", "https://api.openai.com/v1")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("SLOP_API_KEY", "")
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("SLOP_CHAT_MODEL", "gpt-4o")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("SLOP_EMBED_MODEL", "text-embedding-3-small")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SLOP_MAX_TOKENS", "1024"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("SLOP_TEMPERATURE", "0.9"))
    )

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                "No API key found. Set SLOP_API_KEY environment variable or pass api_key to Config."
            )

    def openai_client(self):
        """Return an openai.OpenAI client configured for this endpoint."""
        import openai  # noqa: PLC0415
        return openai.OpenAI(base_url=self.api_base_url, api_key=self.api_key)
