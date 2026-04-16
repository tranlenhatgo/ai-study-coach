from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Quiz API
    quiz_api_url: str = "http://localhost:8080"

    # Ollama (local LLM)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    ollama_timeout: int = 120

    # External LLM (fallback for heavy tasks)
    external_llm_provider: str = "groq"  # "groq" or "google"
    external_llm_api_key: str = ""
    external_llm_model: str = "llama-3.1-8b-instant"

    # Database
    database_url: str = "sqlite+aiosqlite:///./study_coach.db"

    # App
    app_name: str = "AI Study Coach"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    model_config = {"env_file": ".env", "env_prefix": "COACH_"}

settings = Settings()
