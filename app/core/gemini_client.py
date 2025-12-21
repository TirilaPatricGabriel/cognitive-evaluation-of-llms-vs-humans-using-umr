import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.GEMINI_API_KEY)


def call_gemini(prompt: str, model_name: str = None) -> str:
    if model_name is None:
        model_name = settings.MODEL

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    # Handle safety blocks and other response issues
    try:
        return response.text
    except ValueError as e:
        # Check if blocked by safety filters
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            # finish_reason: 1=SAFETY, 2=MAX_TOKENS, 3=RECITATION, 4=OTHER
            if finish_reason == 1:
                error_msg = "[SAFETY_BLOCKED] Content generation blocked by safety filters"
            elif finish_reason == 2:
                error_msg = "[MAX_TOKENS] Response exceeded maximum token length"
            elif finish_reason == 3:
                error_msg = "[RECITATION] Content blocked due to recitation concerns"
            else:
                error_msg = f"[ERROR] Generation failed with finish_reason={finish_reason}"

            # Log the issue but return a placeholder so pipeline continues
            print(f"WARNING: {error_msg}")
            return error_msg
        else:
            # Unknown error, re-raise
            raise
