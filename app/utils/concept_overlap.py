import logging
import re
from typing import Dict, List, Set
import penman

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptOverlapCalculator:

    def extract_concepts(self, umr_str: str) -> Set[str]:
        try:
            graph = penman.decode(umr_str)
            concepts = set()

            for triple in graph.instances():
                concept = triple.target
                if concept:
                    concepts.add(str(concept))

            return concepts

        except Exception as e:
            logger.warning(f"Failed to extract concepts: {e}")
            return set()

    def calculate(self, human_umr: str, llm_umr: str) -> Dict:
        if not human_umr or not human_umr.strip() or not llm_umr or not llm_umr.strip():
            return {
                "jaccard": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "shared_concepts": [],
                "human_only": [],
                "llm_only": [],
                "human_count": 0,
                "llm_count": 0,
                "shared_count": 0
            }

        human_concepts = self.extract_concepts(human_umr)
        llm_concepts = self.extract_concepts(llm_umr)

        if not human_concepts and not llm_concepts:
            return {
                "jaccard": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "shared_concepts": [],
                "human_only": [],
                "llm_only": [],
                "human_count": 0,
                "llm_count": 0,
                "shared_count": 0
            }

        intersection = human_concepts & llm_concepts
        union = human_concepts | llm_concepts

        jaccard = len(intersection) / len(union) if union else 0.0
        precision = len(intersection) / len(llm_concepts) if llm_concepts else 0.0
        recall = len(intersection) / len(human_concepts) if human_concepts else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "jaccard": round(jaccard, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "shared_concepts": sorted(list(intersection)),
            "human_only": sorted(list(human_concepts - llm_concepts)),
            "llm_only": sorted(list(llm_concepts - human_concepts)),
            "human_count": len(human_concepts),
            "llm_count": len(llm_concepts),
            "shared_count": len(intersection)
        }


def create_concept_overlap_calculator() -> ConceptOverlapCalculator:
    return ConceptOverlapCalculator()
