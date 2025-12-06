from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = 'Cognitive Evaluation of LLMs vs Humans using UMR'
    VERSION: str = '0.1.0'

    GEMINI_API_KEY: str = Field(..., description="Gemini API Key")
    GOOGLE_API_KEY: str = Field(..., description="Google API Key")

    MODEL: str = Field("gemini-2.5-flash", description="Basic model for simple tasks")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

@lru_cache
def get_settings():
    return Settings()