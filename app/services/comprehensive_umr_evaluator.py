import logging
from typing import Dict, List, Any
from app.utils.bertscore_calculator import create_bertscore_calculator
from app.utils.soft_smatch import create_soft_smatch_calculator
from app.utils.concept_overlap import create_concept_overlap_calculator
from app.utils.complexity_similarity import create_complexity_similarity_analyzer
from app.services.umr_analyzer import create_umr_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveUMREvaluator:

    def __init__(self):
        logger.info("Initializing Comprehensive UMR Evaluator")
        self.bertscore_calculator = create_bertscore_calculator()
        self.soft_smatch_calculator = create_soft_smatch_calculator()
        self.concept_overlap_calculator = create_concept_overlap_calculator()
        self.complexity_analyzer = create_complexity_similarity_analyzer()
        self.umr_analyzer = create_umr_analyzer()
        logger.info("All evaluators initialized successfully")

    def evaluate_pair(self,
                      human_text: str,
                      llm_text: str,
                      human_umr: str,
                      llm_umr: str,
                      text_id: str,
                      language: str = "romanian") -> Dict[str, Any]:

        logger.info(f"Evaluating pair: {text_id}")

        bertscore = self.bertscore_calculator.calculate_single(human_text, llm_text, language)

        standard_smatch = self.umr_analyzer.calculate_smatch(human_umr, llm_umr, use_official=True)

        soft_smatch = self.soft_smatch_calculator.calculate(human_umr, llm_umr, threshold=0.75)

        concept_overlap = self.concept_overlap_calculator.calculate(human_umr, llm_umr)

        human_complexity = self.umr_analyzer.analyze_complexity(human_umr)
        llm_complexity = self.umr_analyzer.analyze_complexity(llm_umr)
        complexity_similarity = self.complexity_analyzer.calculate(human_complexity, llm_complexity)

        results = {
            "text_id": text_id,
            "language": language,

            "bertscore": bertscore,

            "standard_smatch": standard_smatch,

            "soft_smatch": soft_smatch,

            "concept_overlap": concept_overlap,

            "human_complexity": human_complexity,
            "llm_complexity": llm_complexity,
            "complexity_similarity": complexity_similarity
        }

        results["overall_assessment"] = self._assess_overall(results)

        logger.info(f"Evaluation complete for {text_id}: Overall Score = {results['overall_assessment']['overall_score']:.3f}")

        return results

    def _assess_overall(self, results: Dict) -> Dict[str, Any]:
        text_score = results['bertscore'].get('f1', 0.0)

        umr_score = max(
            results['soft_smatch'].get('f1', 0.0),
            results['concept_overlap'].get('f1', 0.0)
        )

        structure_score = results['complexity_similarity'].get('overall_score', 0.0)

        doc_score = 0.0

        overall_score = (
            text_score * 0.25 +
            umr_score * 0.50 +
            structure_score * 0.25 +
            doc_score * 0.0
        )

        if overall_score > 0.80:
            interpretation = "Excellent semantic preservation"
        elif overall_score > 0.70:
            interpretation = "Good semantic preservation"
        elif overall_score > 0.60:
            interpretation = "Acceptable semantic preservation"
        elif overall_score > 0.50:
            interpretation = "Moderate semantic preservation"
        else:
            interpretation = "Significant semantic divergence"

        recommendation = self._generate_recommendation(results)

        return {
            "overall_score": round(overall_score, 4),
            "interpretation": interpretation,
            "text_level_score": round(text_score, 4),
            "umr_level_score": round(umr_score, 4),
            "structure_level_score": round(structure_score, 4),
            "document_level_score": round(doc_score, 4),
            "recommendation": recommendation
        }

    def _generate_recommendation(self, results: Dict) -> str:
        issues = []

        if results['bertscore'].get('f1', 0.0) < 0.80:
            issues.append("Text generation quality needs improvement")

        if results['soft_smatch'].get('f1', 0.0) < 0.50:
            issues.append("Concept-level alignment is poor - check vocabulary drift")

        node_ratio = results['complexity_similarity'].get('node_ratio', 1.0)
        if node_ratio < 0.7 or node_ratio > 1.5:
            issues.append("UMR parser consistency issue - node count varies too much")

        reentrancy_ratio = results['complexity_similarity'].get('reentrancy_ratio', 1.0)
        if reentrancy_ratio > 2.0:
            issues.append("LLM text has excessive connections - may be over-elaborating")

        if not issues:
            return "No major issues detected. Pipeline working well."
        else:
            return "Issues found: " + "; ".join(issues)

    def evaluate_corpus(self,
                        human_graphs: List[Dict],
                        llm_graphs: List[Dict],
                        human_texts: List[str] = None,
                        llm_texts: List[str] = None) -> List[Dict]:

        if len(human_graphs) != len(llm_graphs):
            logger.error(f"Mismatch: {len(human_graphs)} human graphs vs {len(llm_graphs)} LLM graphs")
            return []

        logger.info(f"Evaluating {len(human_graphs)} text pairs")

        results = []

        for idx, (h_data, l_data) in enumerate(zip(human_graphs, llm_graphs)):
            h_umr = h_data.get('umr_graph', '')
            l_umr = l_data.get('umr_graph', '')

            h_text = h_data.get('original_text', '') if not human_texts else human_texts[idx]
            l_text = l_data.get('original_text', '') if not llm_texts else llm_texts[idx]

            text_id = h_data.get('filename') or h_data.get('subcategory', f'text_{idx}')
            language = h_data.get('language', 'romanian')

            result = self.evaluate_pair(h_text, l_text, h_umr, l_umr, text_id, language)
            result['subcategory'] = h_data.get('subcategory', '')

            results.append(result)

        return results

    def compute_aggregate_statistics(self, comprehensive_results: List[Dict]) -> Dict[str, Any]:
        if not comprehensive_results:
            return {}

        total = len(comprehensive_results)

        avg_bertscore_f1 = sum(r['bertscore']['f1'] for r in comprehensive_results) / total
        avg_soft_smatch_f1 = sum(r['soft_smatch']['f1'] for r in comprehensive_results) / total
        avg_concept_overlap_f1 = sum(r['concept_overlap']['f1'] for r in comprehensive_results) / total
        avg_standard_smatch_f1 = sum(r['standard_smatch']['smatch_f1'] for r in comprehensive_results) / total
        avg_complexity_score = sum(r['complexity_similarity']['overall_score'] for r in comprehensive_results) / total
        avg_overall_score = sum(r['overall_assessment']['overall_score'] for r in comprehensive_results) / total

        texts_excellent = sum(1 for r in comprehensive_results if r['overall_assessment']['overall_score'] > 0.80)
        texts_good = sum(1 for r in comprehensive_results if 0.70 <= r['overall_assessment']['overall_score'] <= 0.80)
        texts_acceptable = sum(1 for r in comprehensive_results if 0.60 <= r['overall_assessment']['overall_score'] < 0.70)
        texts_poor = sum(1 for r in comprehensive_results if r['overall_assessment']['overall_score'] < 0.60)

        if avg_overall_score > 0.75:
            final_interpretation = "Excellent - Pipeline validated, ready for publication"
        elif avg_overall_score > 0.65:
            final_interpretation = "Good - Minor improvements needed"
        elif avg_overall_score > 0.55:
            final_interpretation = "Acceptable - Moderate improvements needed"
        else:
            final_interpretation = "Needs improvement - Significant issues detected"

        return {
            "total_pairs": total,
            "average_bertscore_f1": round(avg_bertscore_f1, 4),
            "average_soft_smatch_f1": round(avg_soft_smatch_f1, 4),
            "average_concept_overlap_f1": round(avg_concept_overlap_f1, 4),
            "average_standard_smatch_f1": round(avg_standard_smatch_f1, 4),
            "average_complexity_score": round(avg_complexity_score, 4),
            "average_overall_score": round(avg_overall_score, 4),
            "distribution": {
                "excellent": int(texts_excellent),
                "good": int(texts_good),
                "acceptable": int(texts_acceptable),
                "poor": int(texts_poor)
            },
            "interpretation": final_interpretation,
            "success_criteria": {
                "bertscore_target": 0.85,
                "bertscore_achieved": bool(avg_bertscore_f1 >= 0.85),
                "soft_smatch_target": 0.50,
                "soft_smatch_achieved": bool(avg_soft_smatch_f1 >= 0.50),
                "overall_target": 0.70,
                "overall_achieved": bool(avg_overall_score >= 0.70)
            }
        }


def create_comprehensive_evaluator() -> ComprehensiveUMREvaluator:
    return ComprehensiveUMREvaluator()
