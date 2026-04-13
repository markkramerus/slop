"""
config.py — API client configuration for any OpenAI-compatible endpoint.

Set via environment variables or pass a Config object explicitly.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM/Chat API configuration
    api_base_url: str = field(
        default_factory=lambda: os.getenv("GENERATOR_API_BASE_URL", "https://api.openai.com/v1")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("GENERATOR_API_KEY", "")
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("GENERATOR_CHAT_MODEL", "gpt-4o")
    )

    # Embedding API configuration
    embed_api_base_url: str = field(
        default_factory=lambda: os.getenv("EMBED_API_BASE_URL", "https://api.openai.com/v1")
    )
    embed_api_key: str = field(
        default_factory=lambda: os.getenv("EMBED_API_KEY", "")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "text-embedding-3-small")
    )
    
    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                "No API key found. Set GENERATOR_API_KEY environment variable or pass api_key to Config."
            )
        if not self.embed_api_key:
            raise ValueError(
                "No embedding API key found. Set EMBED_API_KEY environment variable or pass embed_api_key to Config."
            )

    def openai_client(self):
        """Return an openai.OpenAI client configured for chat/generation."""
        import openai  # noqa: PLC0415
        return openai.OpenAI(base_url=self.api_base_url, api_key=self.api_key)
    
    def async_openai_client(self):
        """Return an async openai.AsyncOpenAI client configured for chat/generation."""
        import openai  # noqa: PLC0415
        return openai.AsyncOpenAI(base_url=self.api_base_url, api_key=self.api_key)
    
    def embedding_client(self):
        """Return an openai.OpenAI client configured for embeddings."""
        import openai  # noqa: PLC0415
        return openai.OpenAI(base_url=self.embed_api_base_url, api_key=self.embed_api_key)
    
    def async_embedding_client(self):
        """Return an async openai.AsyncOpenAI client configured for embeddings."""
        import openai  # noqa: PLC0415
        return openai.AsyncOpenAI(base_url=self.embed_api_base_url, api_key=self.embed_api_key)
