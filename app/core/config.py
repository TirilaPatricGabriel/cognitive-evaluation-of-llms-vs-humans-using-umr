from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = 'Cognitive Evaluation of LLMs vs Humans using UMR'
    VERSION: str = '0.1.0'

    GEMINI_API_KEY: str = Field(..., description="Gemini API Key")
    GOOGLE_API_KEY: str = Field(..., description="Google API Key")

    MODEL: str = Field("gemini-3-flash-preview", description="Gemini 3 Flash for complex reasoning")
    UMR_THINKING_LEVEL: str = Field("LOW", description="Thinking level for UMR parsing: MINIMAL, LOW, MEDIUM, HIGH")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

@lru_cache
def get_settings():
    return Settings()