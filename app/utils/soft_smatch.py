import logging
from typing import Dict, List
import penman
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftSmatchCalculator:

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def extract_concepts(self, umr_str: str) -> List[str]:
        if not umr_str or not umr_str.strip():
            return []

        concepts = []

        graph_strings = umr_str.strip().split('\n\n')

        for graph_str in graph_strings:
            graph_str = graph_str.strip()
            if not graph_str:
                continue

            try:
                graph = penman.decode(graph_str)
                for triple in graph.instances():
                    concept = triple.target
                    if concept:
                        concepts.append(str(concept))
            except Exception as e:
                logger.debug(f"Failed to parse one graph segment: {e}")
                continue

        return concepts

    def calculate(self, human_umr: str, llm_umr: str, threshold: float = 0.75) -> Dict:
        if not human_umr or not human_umr.strip() or not llm_umr or not llm_umr.strip():
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "matched_pairs": 0,
                "human_concepts": 0,
                "llm_concepts": 0,
                "threshold": threshold
            }

        human_concepts = self.extract_concepts(human_umr)
        llm_concepts = self.extract_concepts(llm_umr)

        if not human_concepts and not llm_concepts:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "matched_pairs": 0,
                "human_concepts": 0,
                "llm_concepts": 0,
                "threshold": threshold
            }

        if not llm_concepts:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "matched_pairs": 0,
                "human_concepts": len(human_concepts),
                "llm_concepts": 0,
                "threshold": threshold
            }

        if not human_concepts:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "matched_pairs": 0,
                "human_concepts": 0,
                "llm_concepts": len(llm_concepts),
                "threshold": threshold
            }

        try:
            human_embeddings = self.model.encode(human_concepts, convert_to_numpy=True)
            llm_embeddings = self.model.encode(llm_concepts, convert_to_numpy=True)

            similarity_matrix = np.dot(human_embeddings, llm_embeddings.T)
            norms_human = np.linalg.norm(human_embeddings, axis=1, keepdims=True)
            norms_llm = np.linalg.norm(llm_embeddings, axis=1, keepdims=True)
            similarity_matrix = similarity_matrix / (norms_human @ norms_llm.T)

            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

            matches = sum(similarity_matrix[i, j] >= threshold for i, j in zip(row_ind, col_ind))

            precision = matches / len(llm_concepts) if llm_concepts else 0.0
            recall = matches / len(human_concepts) if human_concepts else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            return {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "matched_pairs": int(matches),
                "human_concepts": len(human_concepts),
                "llm_concepts": len(llm_concepts),
                "threshold": threshold,
                "avg_similarity": round(float(np.mean([similarity_matrix[i, j] for i, j in zip(row_ind, col_ind)])), 4)
            }

        except Exception as e:
            logger.error(f"Soft-Smatch calculation failed: {e}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "matched_pairs": 0,
                "human_concepts": len(human_concepts),
                "llm_concepts": len(llm_concepts),
                "threshold": threshold,
                "error": str(e)
            }


def create_soft_smatch_calculator() -> SoftSmatchCalculator:
    return SoftSmatchCalculator()
