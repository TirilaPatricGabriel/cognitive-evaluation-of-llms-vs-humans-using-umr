import json
import re
import time
from typing import Dict, List, Optional
from app.core.gemini_client import call_gemini
from app.utils.load_yaml_prompts import load_prompt

class UMRParser:
    def __init__(self):
        self.config_en = self._load_config("app/prompts/umr_config_en.yaml")
        self.config_ro = self._load_config("app/prompts/umr_config_ro.yaml")

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
        prompt_parts.append("\nRemember: Return ONLY a valid JSON object with the key 'umr_graph'. Ensure all quotes within the graph string are escaped.")
        
        return "\n".join(prompt_parts)

    def _extract_json(self, response: str) -> Optional[Dict]:
        text = response.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
            if "umr_graph" in parsed:
                parsed["umr_graph"] = self._clean_graph(parsed["umr_graph"])
            return parsed
        except json.JSONDecodeError:
            pass

        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "umr_graph" in parsed:
                    parsed["umr_graph"] = self._clean_graph(parsed["umr_graph"])
                return parsed
        except json.JSONDecodeError:
            pass

        graph_match = re.search(r':\s*"((\(.*\))|(\(.*\n.*))"', text, re.DOTALL)
        if graph_match:
            graph_content = graph_match.group(1)
            clean_graph = graph_content.replace('\\"', '"').replace('\\n', '\n')
            return {"umr_graph": self._clean_graph(clean_graph)}

        lisp_match = re.search(r'(\([a-z0-9_]+.*?\))', text, re.DOTALL)
        if lisp_match:
            raw_graph = lisp_match.group(1)

            if raw_graph.count('(') == raw_graph.count(')'):
                clean_graph = raw_graph.replace('\\"', '"').replace('\\n', '\n')
                return {"umr_graph": self._clean_graph(clean_graph)}

        return None

    def _clean_graph(self, graph: str) -> str:
        lines = []
        for line in graph.split('\n'):
            comment_pos = line.find(';')
            if comment_pos != -1:
                line = line[:comment_pos].rstrip()
            if line.strip():
                lines.append(line)
        return '\n'.join(lines)

    def parse_text(self, text: str, language: str, max_retries: int = 3) -> Dict:
        prompt = self._build_prompt(text, language)
        
        for attempt in range(max_retries):
            try:
                response = call_gemini(prompt)
                parsed_json = self._extract_json(response)

                if parsed_json and "umr_graph" in parsed_json:
                    return {
                        "umr_graph": parsed_json["umr_graph"],
                        "success": True,
                        "raw_response": response
                    }
                
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                
                return {
                    "umr_graph": response,
                    "success": False,
                    "raw_response": response,
                    "error": "Failed to parse JSON response after retries"
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                
                return {
                    "umr_graph": "",
                    "success": False,
                    "raw_response": str(e),
                    "error": f"API Parsing Error: {str(e)}"
                }

    def parse_batch(self, texts: List[Dict]) -> List[Dict]:
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
    return UMRParser()