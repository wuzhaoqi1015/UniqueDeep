#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/web_api.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: 基于 FastAPI 的 Agent Web API，提供用于浏览器集成的 SSE 端点。
'''

"""
FastAPI Web API for LangChain Skills Agent.

This module exposes a lightweight SSE bridge so the existing CLI streaming
experience can be rendered in a browser.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import LangChainSkillsAgent, check_api_credentials


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class AgentLike(Protocol):
    """Minimal surface required by the Web API."""

    def get_discovered_skills(self) -> list[dict[str, Any]]: ...

    def get_system_prompt(self) -> str: ...

    def stream_events(
        self, message: str, thread_id: str = "default"
    ) -> Iterator[dict[str, Any]]: ...


_AGENT_SINGLETON: LangChainSkillsAgent | None = None


def _to_sse_frame(event_type: str, payload: dict[str, Any]) -> str:
    """Encode one SSE frame."""
    # "error" conflicts with EventSource transport-level error events in browsers.
    # Use a dedicated SSE event name while keeping payload.type = "error".
    sse_event = "agent_error" if event_type == "error" else event_type
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {sse_event}\ndata: {data}\n\n"


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Parse comma-separated origins from env."""
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)

    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or list(DEFAULT_CORS_ORIGINS)


def _default_agent_provider() -> LangChainSkillsAgent:
    """Lazily initialize a single agent instance for API requests."""
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = LangChainSkillsAgent()
    return _AGENT_SINGLETON


def create_app(agent_provider: Callable[[], AgentLike] | None = None) -> FastAPI:
    """Create FastAPI app with injectable agent provider (for tests)."""
    provider = agent_provider or _default_agent_provider

    app = FastAPI(
        title="LangChain Skills Agent Web API",
        version="0.1.0",
        description="SSE bridge for stream_events()",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(os.getenv("SKILLS_WEB_CORS_ORIGINS")),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_credentials_configured": check_api_credentials(),
        }

    @app.get("/api/skills")
    def list_skills() -> dict[str, Any]:
        agent = provider()
        return {"skills": agent.get_discovered_skills()}

    @app.get("/api/prompt")
    def get_prompt() -> dict[str, str]:
        agent = provider()
        return {"prompt": agent.get_system_prompt()}

    @app.get("/api/chat/stream")
    def chat_stream(
        message: str = Query(..., min_length=1),
        thread_id: str = Query("default", min_length=1),
    ) -> StreamingResponse:
        def event_stream() -> Iterator[str]:
            error_emitted = False
            try:
                agent = provider()
            except Exception as exc:  # pragma: no cover - defensive path
                payload = {
                    "type": "error",
                    "message": f"Failed to initialize agent: {exc}",
                }
                yield _to_sse_frame("error", payload)
                return

            try:
                for event in agent.stream_events(message, thread_id=thread_id):
                    event_type = str(event.get("type", "message"))
                    if event_type == "error":
                        error_emitted = True
                    yield _to_sse_frame(event_type, event)
            except GeneratorExit:
                return
            except Exception as exc:
                if not error_emitted:
                    payload = {"type": "error", "message": str(exc)}
                    yield _to_sse_frame("error", payload)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


def main() -> None:
    """Run development server for the Web API."""
    import uvicorn

    host = os.getenv("SKILLS_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("SKILLS_WEB_PORT", "8000"))
    reload_enabled = os.getenv("SKILLS_WEB_RELOAD", "").lower() in ("1", "true", "yes")

    uvicorn.run(
        "uniquedeep.web_api:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


__all__ = [
    "app",
    "create_app",
    "main",
]
