import json
import re
import time
import logging
from typing import Dict, List, Optional
from app.core.gemini_client import call_gemini
from app.core.config import get_settings
from app.utils.umr_validator import UMRValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


class UMRParser:

    def __init__(self):
        self.config_en = self._load_config("app/prompts/umr_config_en.yaml")
        self.config_ro = self._load_config("app/prompts/umr_config_ro.yaml")
        self.validator = UMRValidator()

    def _load_config(self, config_path: str) -> Dict:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_prompt(self, text: str, language: str) -> str:
        config = self.config_en if language == "english" else self.config_ro
        system_prompt = config['system_prompt']
        examples = config.get('few_shot_examples', [])

        prompt_parts = [system_prompt]

        if examples:
            prompt_parts.append("\n### EXAMPLES:\n")
            for example in examples:
                for key, ex_data in example.items():
                    prompt_parts.append(f"\nInput: {ex_data['input']}")
                    prompt_parts.append(f"Output: {ex_data['output']}")

        prompt_parts.append(f"\n### NOW PARSE THIS TEXT:\n{text}")
        prompt_parts.append("\n" + "="*70)
        prompt_parts.append("\nBEFORE RETURNING YOUR OUTPUT, VERIFY:")
        prompt_parts.append("1. Every event concept has :aspect annotation (State, Activity, Performance, Habitual, Endeavor)")
        prompt_parts.append("2. NO accomplishment or achievement aspects - use Performance instead")
        prompt_parts.append("3. NO be-01 or be-02 - use abstract concepts (-91) for copula")
        prompt_parts.append("4. Multi-word names use separate :op1, :op2, :op3")
        prompt_parts.append("5. All parentheses are balanced")
        prompt_parts.append("6. Output is valid JSON with 'sentences' array and 'doc_level' object")
        if language == "romanian":
            prompt_parts.append("7. Romanian verbs use -00 suffix (not English PropBank frames)")
            prompt_parts.append("8. Romanian verbs use generic roles (:actor, :undergoer) not :ARG0/:ARG1")
        prompt_parts.append("="*70)

        return "\n".join(prompt_parts)

    def _build_prompt_with_feedback(self, text: str, language: str, failed_output: str, errors: List[str]) -> str:
        config = self.config_en if language == "english" else self.config_ro
        system_prompt = config['system_prompt']

        prompt_parts = [system_prompt]
        prompt_parts.append("\n" + "!"*70)
        prompt_parts.append("PREVIOUS ATTEMPT FAILED VALIDATION!")
        prompt_parts.append("\nYour previous output had these CRITICAL errors:")
        for i, error in enumerate(errors[:5], 1):
            prompt_parts.append(f"  {i}. {error}")

        prompt_parts.append("\n--- Your failed output (first 500 chars) ---")
        prompt_parts.append(failed_output[:500] + "..." if len(failed_output) > 500 else failed_output)
        prompt_parts.append("--- End of failed output ---")

        prompt_parts.append("\nFIX THESE ERRORS and generate a VALID UMR output.")
        prompt_parts.append("!"*70 + "\n")

        prompt_parts.append(f"\n### TEXT TO PARSE:\n{text}")
        prompt_parts.append("\n" + "="*70)
        prompt_parts.append("\nCRITICAL REQUIREMENTS:")
        prompt_parts.append("  1. EVERY event node MUST have :aspect (State, Activity, Performance, Habitual, Endeavor)")
        prompt_parts.append("  2. DO NOT use accomplishment or achievement - use Performance")
        prompt_parts.append("  3. DO NOT use be-01 or be-02 - use have-role-91, have-mod-91, exist-91")
        prompt_parts.append("  4. Multi-word names: use :op1 \"Word1\" :op2 \"Word2\"")
        prompt_parts.append("  5. Balanced parentheses")
        prompt_parts.append("  6. Valid JSON with 'sentences' and 'doc_level'")
        if language == "romanian":
            prompt_parts.append("  7. Romanian: Use native lemmas with -00 (not English frames)")
            prompt_parts.append("  8. Romanian: Use :actor/:undergoer (not :ARG0/:ARG1 for verbs)")
        prompt_parts.append("="*70)

        return "\n".join(prompt_parts)

    def _extract_json(self, response: str) -> Optional[Dict]:
        text = response.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        try:
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError:
            pass

        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
        except json.JSONDecodeError:
            pass

        try:
            fixed_text = re.sub(r',\s*([}\]])', r'\1', text)
            fixed_text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)

            json_match = re.search(r'\{.*\}', fixed_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                logger.info("JSON extracted after fixing common issues")
                return parsed
        except json.JSONDecodeError:
            pass

        # Last resort: try to extract sentences array directly
        try:
            sentences_match = re.search(r'"sentences"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if sentences_match:
                sentences_str = sentences_match.group(0)
                minimal_json = '{' + sentences_str + ', "doc_level": {"temporal": [], "modal": [], "coref": []}}'
                parsed = json.loads(minimal_json)
                logger.info("JSON extracted using minimal reconstruction")
                return parsed
        except Exception:
            pass

        return None

    def _convert_legacy_format(self, parsed: Dict) -> Dict:
        if "sentences" in parsed:
            return parsed

        if "umr_graph" in parsed:
            return {
                "sentences": [
                    {"id": "s1", "graph": parsed["umr_graph"]}
                ],
                "doc_level": {
                    "temporal": [],
                    "modal": ["(AUTH :FullAff s1)"],
                    "coref": []
                }
            }

        return parsed

    def parse_text(self, text: str, language: str, max_retries: int = 2) -> Dict:
        prompt = self._build_prompt(text, language)
        thinking_level = settings.UMR_THINKING_LEVEL

        logger.info(f"Parsing text ({len(text)} chars) in {language} with thinking_level={thinking_level}")

        for attempt in range(max_retries):
            try:
                response = call_gemini(prompt, thinking_level=thinking_level, temperature=0.3)

                if response.startswith("[") and ("ERROR" in response or "SAFETY" in response or "RATE_LIMIT" in response):
                    logger.warning(f"API error on attempt {attempt + 1}: {response[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {
                        "sentences": [],
                        "doc_level": {"temporal": [], "modal": [], "coref": []},
                        "success": False,
                        "raw_response": response,
                        "error": response,
                        "validation": None
                    }

                parsed_json = self._extract_json(response)

                if not parsed_json:
                    logger.warning(f"Failed to extract JSON on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return {
                        "sentences": [],
                        "doc_level": {"temporal": [], "modal": [], "coref": []},
                        "success": False,
                        "raw_response": response,
                        "error": "Failed to extract JSON from response",
                        "validation": None
                    }

                parsed_json = self._convert_legacy_format(parsed_json)

                validation = self.validator.validate_umr_output(parsed_json, language)

                logger.info(f"UMR output extracted. Valid: {validation['valid']}, Quality: {validation['quality_score']}")

                if validation['errors']:
                    logger.warning(f"Validation errors: {validation['errors']}")

                    # Try auto-repair before retrying with LLM
                    if any("missing mandatory :aspect" in err for err in validation['errors']):
                        logger.info("Attempting auto-repair for missing :aspect annotations...")
                        repaired_json = self.validator.repair_umr_output(parsed_json, language)
                        repair_validation = self.validator.validate_umr_output(repaired_json, language)

                        if repair_validation['valid'] or len(repair_validation['errors']) < len(validation['errors']):
                            logger.info(f"Auto-repair successful. Errors reduced from {len(validation['errors'])} to {len(repair_validation['errors'])}")
                            parsed_json = repaired_json
                            validation = repair_validation

                    if validation['errors'] and attempt < max_retries - 1:
                        logger.warning(f"Validation FAILED on attempt {attempt + 1}. Retrying with error feedback...")
                        prompt = self._build_prompt_with_feedback(text, language, response, validation['errors'])
                        time.sleep(1 * (attempt + 1))
                        continue
                    elif validation['errors']:
                        logger.error(f"Validation FAILED on final attempt {attempt + 1}.")
                        # Try one more auto-repair on final failure
                        repaired_json = self.validator.repair_umr_output(parsed_json, language)
                        return {
                            "sentences": repaired_json.get("sentences", []),
                            "doc_level": repaired_json.get("doc_level", {"temporal": [], "modal": [], "coref": []}),
                            "success": False,
                            "raw_response": response,
                            "error": f"Validation failed after {max_retries} attempts: {validation['errors'][0]}",
                            "validation": validation
                        }

                if validation['warnings']:
                    logger.info(f"Validation warnings (first 3): {validation['warnings'][:3]}")

                return {
                    "sentences": parsed_json.get("sentences", []),
                    "doc_level": parsed_json.get("doc_level", {"temporal": [], "modal": [], "coref": []}),
                    "success": True,
                    "raw_response": response,
                    "validation": validation
                }

            except Exception as e:
                logger.error(f"Exception on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue

                return {
                    "sentences": [],
                    "doc_level": {"temporal": [], "modal": [], "coref": []},
                    "success": False,
                    "raw_response": str(e),
                    "error": f"API Parsing Error: {str(e)}",
                    "validation": None
                }

        return {
            "sentences": [],
            "doc_level": {"temporal": [], "modal": [], "coref": []},
            "success": False,
            "raw_response": "",
            "error": "Max retries exceeded",
            "validation": None
        }

    def get_combined_graph(self, result: Dict) -> str:
        sentences = result.get("sentences", [])
        if not sentences:
            return ""

        if len(sentences) == 1:
            return sentences[0].get("graph", "")

        graphs = []
        for sent in sentences:
            graph = sent.get("graph", "")
            if graph:
                graphs.append(graph)

        if not graphs:
            return ""

        return "\n\n".join(graphs)

    def parse_batch(self, texts: List[Dict]) -> List[Dict]:
        results = []
        for item in texts:
            result = self.parse_text(item['text'], item['language'])
            result.update({
                'original_text': item['text'],
                'language': item['language'],
                'metadata': item.get('metadata', {}),
                'umr_graph': self.get_combined_graph(result)
            })
            results.append(result)
        return results


def create_umr_parser() -> UMRParser:
    return UMRParser()
