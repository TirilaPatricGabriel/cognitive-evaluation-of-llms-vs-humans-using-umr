import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.GEMINI_API_KEY)


def call_gemini(prompt: str, model_name: str = None) -> str:
    if model_name is None:
        model_name = settings.MODEL

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text
