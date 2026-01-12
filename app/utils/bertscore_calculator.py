import logging
from typing import List, Dict
from bert_score import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTScoreCalculator:

    def __init__(self):
        self.supported_languages = {
            "english": "en",
            "romanian": "ro"
        }

    def calculate(self, reference_texts: List[str], candidate_texts: List[str], language: str = "romanian") -> Dict:
        lang_code = self.supported_languages.get(language, "ro")

        logger.info(f"Calculating BERTScore for {len(reference_texts)} text pairs in {language} (code: {lang_code})")

        try:
            P, R, F1 = score(
                candidate_texts,
                reference_texts,
                lang=lang_code,
                verbose=False,
                rescale_with_baseline=True
            )

            result = {
                "precision": round(P.mean().item(), 4),
                "recall": round(R.mean().item(), 4),
                "f1": round(F1.mean().item(), 4),
                "precision_scores": [round(p.item(), 4) for p in P],
                "recall_scores": [round(r.item(), 4) for r in R],
                "f1_scores": [round(f.item(), 4) for f in F1],
                "num_pairs": len(reference_texts),
                "language": language
            }

            logger.info(f"BERTScore computed: P={result['precision']}, R={result['recall']}, F1={result['f1']}")

            return result

        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": str(e),
                "num_pairs": len(reference_texts),
                "language": language
            }

    def calculate_single(self, reference_text: str, candidate_text: str, language: str = "romanian") -> Dict:
        result = self.calculate([reference_text], [candidate_text], language)

        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "language": language
        }


def create_bertscore_calculator() -> BERTScoreCalculator:
    return BERTScoreCalculator()
