"""Health check endpoint."""

from fastapi import APIRouter
from server.llm.ollama import ollama_client
from server.llm.external import external_client

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check service health and LLM availability."""
    ollama_ok = await ollama_client.is_available()
    external_ok = await external_client.is_available()

    return {
        "status": "ok" if (ollama_ok or external_ok) else "degraded",
        "ollama": "connected" if ollama_ok else "unavailable",
        "external_llm": "configured" if external_ok else "not configured",
    }
