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
    """
    Research-grade UMR parser with strict validation.

    Design principles:
    1. Enhanced prompts prevent errors (proactive)
    2. Validation catches errors immediately (defensive)
    3. Retry with feedback fixes errors (corrective)
    4. Never return invalid graphs (integrity)
    """

    def __init__(self):
        self.config_en = self._load_config("app/prompts/umr_config_en.yaml")
        self.config_ro = self._load_config("app/prompts/umr_config_ro.yaml")
        self.validator = UMRValidator()

    def _load_config(self, config_path: str) -> Dict:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_prompt(self, text: str, language: str) -> str:
        """Build initial prompt with examples and verification checklist"""
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
        prompt_parts.append("1. Every event concept (ending in -01, -02, -03, etc.) has :aspect annotation")
        prompt_parts.append("2. Multi-word names use separate :op1, :op2, :op3 (NOT multiple strings in one :op)")
        prompt_parts.append("3. All parentheses are balanced")
        prompt_parts.append("4. Output is valid JSON with 'umr_graph' key")
        prompt_parts.append("="*70)

        return "\n".join(prompt_parts)

    def _build_prompt_with_feedback(self, text: str, language: str, failed_graph: str, errors: List[str]) -> str:
        """Build retry prompt with specific error feedback"""
        config = self.config_en if language == "english" else self.config_ro
        system_prompt = config['system_prompt']

        prompt_parts = [system_prompt]
        prompt_parts.append("\n" + "!"*70)
        prompt_parts.append("⚠️  PREVIOUS ATTEMPT FAILED VALIDATION!")
        prompt_parts.append("\nYour previous output had these CRITICAL errors:")
        for i, error in enumerate(errors[:3], 1):
            prompt_parts.append(f"  {i}. {error}")

        prompt_parts.append("\n--- Your failed graph (first 500 chars) ---")
        prompt_parts.append(failed_graph[:500] + "..." if len(failed_graph) > 500 else failed_graph)
        prompt_parts.append("--- End of failed graph ---")

        prompt_parts.append("\n🔧 FIX THESE ERRORS and generate a VALID UMR graph.")
        prompt_parts.append("!"*70 + "\n")

        prompt_parts.append(f"\n### TEXT TO PARSE:\n{text}")
        prompt_parts.append("\n" + "="*70)
        prompt_parts.append("\n✓ CRITICAL REQUIREMENTS (MUST SATISFY ALL):")
        prompt_parts.append("  1. EVERY event node (-01, -02, -03, -04, -05, -11, -91) MUST have :aspect")
        prompt_parts.append("  2. Multi-word names: use :op1 \"Word1\" :op2 \"Word2\" (NOT :op1 \"Word1\" \"Word2\")")
        prompt_parts.append("  3. Every role (:ARG0, :aspect, etc.) MUST have a value immediately after")
        prompt_parts.append("  4. Balanced parentheses (count opening and closing)")
        prompt_parts.append("  5. Valid JSON output with 'umr_graph' key")
        prompt_parts.append("="*70)

        return "\n".join(prompt_parts)

    def _extract_json(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response with multiple fallback strategies"""
        text = response.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Strategy 1: Direct JSON parse
        try:
            parsed = json.loads(text)
            if "umr_graph" in parsed:
                parsed["umr_graph"] = self._clean_graph(parsed["umr_graph"])
            return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object in text
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "umr_graph" in parsed:
                    parsed["umr_graph"] = self._clean_graph(parsed["umr_graph"])
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract graph from key-value pattern
        graph_match = re.search(r':\s*"((\(.*\))|(\(.*\n.*))"', text, re.DOTALL)
        if graph_match:
            graph_content = graph_match.group(1)
            clean_graph = graph_content.replace('\\"', '"').replace('\\n', '\n')
            return {"umr_graph": self._clean_graph(clean_graph)}

        # Strategy 4: Extract raw Lisp-like structure
        lisp_match = re.search(r'(\([a-z0-9_]+.*?\))', text, re.DOTALL)
        if lisp_match:
            raw_graph = lisp_match.group(1)
            if raw_graph.count('(') == raw_graph.count(')'):
                clean_graph = raw_graph.replace('\\"', '"').replace('\\n', '\n')
                return {"umr_graph": self._clean_graph(clean_graph)}

        return None

    def _clean_graph(self, graph: str) -> str:
        """Remove comments and empty lines from graph"""
        lines = []
        for line in graph.split('\n'):
            # Remove inline comments
            comment_pos = line.find(';')
            if comment_pos != -1:
                line = line[:comment_pos].rstrip()
            if line.strip():
                lines.append(line)
        return '\n'.join(lines)

    def parse_text(self, text: str, language: str, max_retries: int = 3) -> Dict:
        """
        Parse text into UMR graph with validation and retry logic.

        Returns Dict with:
            umr_graph: str - The UMR graph in Penman notation
            success: bool - True if validation passed
            raw_response: str - Original LLM response
            validation: Dict - Validation results (errors, warnings, stats)
            error: str (optional) - Error message if parsing failed
        """
        prompt = self._build_prompt(text, language)
        thinking_level = settings.UMR_THINKING_LEVEL

        logger.info(f"Parsing text ({len(text)} chars) in {language} with thinking_level={thinking_level}")

        for attempt in range(max_retries):
            try:
                # Call LLM
                response = call_gemini(prompt, thinking_level=thinking_level)

                # Check for API errors
                if response.startswith("[") and ("ERROR" in response or "SAFETY" in response or "RATE_LIMIT" in response):
                    logger.warning(f"API error on attempt {attempt + 1}: {response[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {
                        "umr_graph": "",
                        "success": False,
                        "raw_response": response,
                        "error": response,
                        "validation": None
                    }

                # Extract JSON
                parsed_json = self._extract_json(response)

                if not parsed_json or "umr_graph" not in parsed_json:
                    logger.warning(f"Failed to extract UMR graph on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return {
                        "umr_graph": response,
                        "success": False,
                        "raw_response": response,
                        "error": "Failed to extract UMR graph from response",
                        "validation": None
                    }

                # Validate graph
                umr_graph = parsed_json["umr_graph"]
                validation = self.validator.validate_graph(umr_graph)

                logger.info(f"UMR graph extracted. Valid: {validation['valid']}, Quality: {validation['quality_score']}")

                # Handle validation errors
                if validation['errors']:
                    logger.warning(f"Validation errors: {validation['errors']}")

                    # Retry with error feedback
                    if attempt < max_retries - 1:
                        logger.warning(f"Graph validation FAILED on attempt {attempt + 1}. Retrying with error feedback...")
                        prompt = self._build_prompt_with_feedback(text, language, umr_graph, validation['errors'])
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        # Last attempt failed - return with error
                        logger.error(f"Graph validation FAILED on final attempt {attempt + 1}. Returning invalid graph.")
                        return {
                            "umr_graph": umr_graph,
                            "success": False,
                            "raw_response": response,
                            "error": f"Validation failed after {max_retries} attempts: {validation['errors'][0]}",
                            "validation": validation
                        }

                # Log warnings (non-blocking)
                if validation['warnings']:
                    logger.info(f"Validation warnings (first 3): {validation['warnings'][:3]}")

                # Success!
                return {
                    "umr_graph": umr_graph,
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
                    "umr_graph": "",
                    "success": False,
                    "raw_response": str(e),
                    "error": f"API Parsing Error: {str(e)}",
                    "validation": None
                }

        # Should never reach here, but defensive programming
        return {
            "umr_graph": "",
            "success": False,
            "raw_response": "",
            "error": "Max retries exceeded",
            "validation": None
        }

    def parse_batch(self, texts: List[Dict]) -> List[Dict]:
        """Parse multiple texts in batch"""
        results = []
        for item in texts:
            result = self.parse_text(item['text'], item['language'])
            result.update({
                'original_text': item['text'],
                'language': item['language'],
                'metadata': item.get('metadata', {})
            })
            results.append(result)
        return results


def create_umr_parser() -> UMRParser:
    """Factory function to create UMR parser instance"""
    return UMRParser()