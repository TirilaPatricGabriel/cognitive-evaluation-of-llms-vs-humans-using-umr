import time
import logging
from google import genai
from google.genai import types
import google.generativeai as genai_legacy
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

client_v3 = genai.Client(api_key=settings.GEMINI_API_KEY)
genai_legacy.configure(api_key=settings.GEMINI_API_KEY)


def is_gemini_3_model(model_name: str) -> bool:
    return model_name.startswith("gemini-3")


def call_gemini(prompt: str, model_name: str = None, thinking_level: str = "LOW", temperature: float = None, max_retries: int = 3) -> str:
    if model_name is None:
        model_name = settings.MODEL

    for attempt in range(max_retries):
        try:
            if is_gemini_3_model(model_name):
                logger.info(f"Using Gemini 3 model: {model_name} with thinking_level={thinking_level}, temperature={temperature}")
                config_params = {
                    "thinking_config": types.ThinkingConfig(thinking_level=thinking_level)
                }
                if temperature is not None:
                    config_params["temperature"] = temperature
                response = client_v3.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_params)
                )
                return response.text
            else:
                logger.info(f"Using legacy Gemini model: {model_name}, temperature={temperature}")
                generation_config = {}
                if temperature is not None:
                    generation_config["temperature"] = temperature
                model = genai_legacy.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config=generation_config if generation_config else None)
                return response.text

        except ValueError as e:
            error_msg = f"[ERROR] Generation failed: {str(e)}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return error_msg

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Rate limit hit")
                if attempt < max_retries - 1:
                    wait_time = 60
                    logger.info(f"Waiting {wait_time} seconds for rate limit...")
                    time.sleep(wait_time)
                else:
                    return f"[RATE_LIMIT] {error_msg}"
            else:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: {error_type}: {error_msg}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return f"[ERROR] {error_type}: {error_msg}"

    return "[ERROR] All retry attempts failed"
