"""AI Study Coach — FastAPI entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.routes import chat, health

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"🚀 {settings.app_name} starting...")
    logger.info(f"   Quiz API: {settings.quiz_api_url}")

    # Check primary LLM: External API
    if settings.external_llm_api_key:
        logger.info(
            f"   ✅ External LLM configured — "
            f"{settings.external_llm_provider}: {settings.external_llm_model}"
        )
    else:
        logger.warning("   ⚠️ No external LLM API key set (COACH_EXTERNAL_LLM_API_KEY)")

    # Check fallback LLM: Ollama (optional)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "?") for m in models]
                logger.info(f"   ✅ Ollama available (fallback) — models: {', '.join(model_names) or 'none'}")
    except Exception:
        logger.info(f"   ℹ️ Ollama not running at {settings.ollama_url} (optional fallback)")

    if not settings.external_llm_api_key:
        logger.warning("   💡 Set COACH_EXTERNAL_LLM_API_KEY in .env to enable the AI coach")

    yield
    logger.info(f"👋 {settings.app_name} shutting down")


app = FastAPI(
    title=settings.app_name,
    description="AI-powered study coach that analyzes quiz performance and creates personalized study plans.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow quiz app frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, tags=["Chat"])
app.include_router(health.router, tags=["Health"])

# Serve widget static files
app.mount("/static", StaticFiles(directory="widget"), name="static")


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
    }
