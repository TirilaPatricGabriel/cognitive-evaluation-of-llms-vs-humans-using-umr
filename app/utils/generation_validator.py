import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationValidator:

    def __init__(self):
        self.max_word_deviation = 0.20
        self.max_sentence_deviation = 0.15
        self.max_node_deviation = 0.30

    def validate_generation(self, human_text: str, llm_text: str,
                           human_nodes: int = None, llm_nodes: int = None) -> Dict:

        human_words = len(human_text.split())
        llm_words = len(llm_text.split())

        human_sentences = len([s for s in human_text.split('.') if s.strip()])
        llm_sentences = len([s for s in llm_text.split('.') if s.strip()])

        word_ratio = llm_words / human_words if human_words > 0 else 0
        sentence_ratio = llm_sentences / human_sentences if human_sentences > 0 else 0

        word_deviation = abs(word_ratio - 1.0)
        sentence_deviation = abs(sentence_ratio - 1.0)

        issues = []
        warnings = []

        if word_deviation > self.max_word_deviation:
            severity = "CRITICAL" if word_deviation > 0.5 else "WARNING"
            msg = f"{severity}: Word count deviation {word_deviation:.1%} ({human_words} -> {llm_words})"
            if severity == "CRITICAL":
                issues.append(msg)
            else:
                warnings.append(msg)

        if sentence_deviation > self.max_sentence_deviation:
            severity = "CRITICAL" if sentence_deviation > 0.3 else "WARNING"
            msg = f"{severity}: Sentence count deviation {sentence_deviation:.1%} ({human_sentences} -> {llm_sentences})"
            if severity == "CRITICAL":
                issues.append(msg)
            else:
                warnings.append(msg)

        if human_nodes and llm_nodes:
            node_ratio = llm_nodes / human_nodes if human_nodes > 0 else 0
            node_deviation = abs(node_ratio - 1.0)

            if node_deviation > self.max_node_deviation:
                severity = "CRITICAL" if node_deviation > 1.0 else "WARNING"
                msg = f"{severity}: UMR node count deviation {node_deviation:.1%} ({human_nodes} -> {llm_nodes})"
                if severity == "CRITICAL":
                    issues.append(msg)
                else:
                    warnings.append(msg)

        quality_score = 1.0 - min(word_deviation + sentence_deviation, 1.0)

        return {
            "valid": len(issues) == 0,
            "quality_score": quality_score,
            "issues": issues,
            "warnings": warnings,
            "metrics": {
                "human_words": human_words,
                "llm_words": llm_words,
                "word_ratio": word_ratio,
                "human_sentences": human_sentences,
                "llm_sentences": llm_sentences,
                "sentence_ratio": sentence_ratio,
                "human_nodes": human_nodes,
                "llm_nodes": llm_nodes,
                "node_ratio": llm_nodes / human_nodes if human_nodes and human_nodes > 0 else None
            }
        }


def create_generation_validator():
    return GenerationValidator()
