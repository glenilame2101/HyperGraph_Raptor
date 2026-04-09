"""Centralized LLM and embedding client factory.

All scripts and modules should import from here instead of
instantiating their own clients. Configuration is read from
environment variables (typically loaded from ``.env``).

Environment variables
---------------------
LLM:
    URL              – OpenAI-compatible base URL  (required)
    MODEL_NAME       – Model identifier            (required)
    OPENAI_API_KEY   – API key                     (required)
    LLM_TEMPERATURE  – Sampling temperature        (default: 0)
    LLM_MAX_TOKENS   – Max generation tokens       (default: 20000)
    LLM_TIMEOUT      – HTTP timeout in seconds     (default: 120)

Embeddings:
    EMBED_URL        – Embedding server base URL   (default: http://127.0.0.1:8080)
    EMBED_MODEL      – Embedding model name        (default: BAAI/bge-m3)
"""
from __future__ import annotations

import logging
import os
import ssl
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Auto-load .env from repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class LocalBGEClient:
    """Embedding client for a local BGE-M3 (or compatible) server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_input_chars: int | None = None,
    ):
        self.base_url = (base_url or os.getenv("EMBED_URL", "http://127.0.0.1:8080")).rstrip("/")
        self.model = model or os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        self.client = httpx.Client(timeout=timeout)
        # Server batch size limits how many tokens can be processed at once.
        # 512 tokens ≈ ~1200 chars (conservative estimate for mixed content).
        self.max_input_chars = max_input_chars or int(os.getenv("EMBED_MAX_CHARS", "1200"))

    def encode(self, text: str, *, max_retries: int = 3) -> np.ndarray:
        log = logging.getLogger(__name__)
        if len(text) > self.max_input_chars:
            log.warning("Truncating embedding input from %d to %d chars",
                        len(text), self.max_input_chars)
            text = text[: self.max_input_chars]
        log.debug("Embedding request: %d chars", len(text))
        for attempt in range(1, max_retries + 1):
            resp = self.client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model, "input": text},
            )
            if resp.status_code == 200:
                return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
            # Token-limit error: halve the text and retry immediately
            if resp.status_code == 500 and "too large to process" in resp.text:
                text = text[: len(text) // 2]
                log.warning("Input too many tokens, shrinking to %d chars (attempt %d/%d)",
                            len(text), attempt, max_retries)
                continue
            if resp.status_code >= 500 and attempt < max_retries:
                wait = 2 ** attempt
                log.warning("Embedding server returned %s, retrying in %ds (attempt %d/%d)",
                            resp.status_code, wait, attempt, max_retries)
                time.sleep(wait)
                continue
            log.error("Embedding failed (HTTP %d), text length: %d chars, "
                      "response: %s", resp.status_code, len(text),
                      resp.text[:500])
            resp.raise_for_status()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _resolve_ssl(cert_path: str | None = None) -> Any:
    """Return an SSL context or True/False for httpx verify."""
    if cert_path and os.path.exists(cert_path):
        return ssl.create_default_context(cafile=cert_path)
    default_cert = _REPO_ROOT / "certs" / "knapp.pem"
    if default_cert.exists():
        return ssl.create_default_context(cafile=str(default_cert))
    return True


def create_llm(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    verify_ssl: bool = True,
    trust_env: bool = True,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance using env vars as defaults.

    Any explicit keyword argument overrides the corresponding env var.
    """
    _base_url = base_url or os.getenv("URL")
    _model = model or os.getenv("MODEL_NAME")
    _api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not _base_url or not _model or not _api_key:
        raise ValueError(
            "Missing LLM configuration. Set URL, MODEL_NAME, and "
            "OPENAI_API_KEY in your .env file or pass them explicitly."
        )

    _temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0"))
    _max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "20000"))
    _timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "120"))

    verify_value = _resolve_ssl() if verify_ssl else False

    http_client = httpx.Client(
        verify=verify_value,
        timeout=_timeout,
        trust_env=trust_env,
    )

    return ChatOpenAI(
        base_url=_base_url,
        model=_model,
        api_key=_api_key,
        http_client=http_client,
        max_tokens=_max_tokens,
        temperature=_temperature,
    )


def create_embed_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
) -> LocalBGEClient:
    """Create a LocalBGEClient using env vars as defaults."""
    return LocalBGEClient(base_url=base_url, model=model, timeout=timeout)
