import logging
import math
from typing import Dict
from scipy.stats import chi2_contingency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexitySimilarityAnalyzer:

    def calculate(self, human_metrics: Dict, llm_metrics: Dict) -> Dict:
        if not human_metrics or not llm_metrics:
            return {
                "overall_score": 0.0,
                "node_ratio": 0.0,
                "node_score": 0.0,
                "depth_delta": 0,
                "depth_score": 0.0,
                "reentrancy_ratio": 0.0,
                "reentrancy_score": 0.0,
                "aspect_p_value": 0.0,
                "interpretation": "No data"
            }

        node_ratio = llm_metrics.get('node_count', 0) / human_metrics.get('node_count', 1) if human_metrics.get('node_count', 0) > 0 else 0.0
        node_score = 1.0 - min(abs(node_ratio - 1.0), 1.0)

        depth_delta = abs(llm_metrics.get('graph_depth', 0) - human_metrics.get('graph_depth', 0))
        depth_score = max(1.0 - (depth_delta / 5.0), 0.0)

        if human_metrics.get('reentrancy_count', 0) > 0:
            reentrancy_ratio = llm_metrics.get('reentrancy_count', 0) / human_metrics.get('reentrancy_count', 1)
            reentrancy_score = 1.0 - min(abs(math.log2(reentrancy_ratio)) if reentrancy_ratio > 0 else 1.0, 1.0)
        else:
            reentrancy_ratio = 0.0
            reentrancy_score = 1.0 if llm_metrics.get('reentrancy_count', 0) == 0 else 0.5

        human_aspects = [
            human_metrics.get('aspect_state_count', 0),
            human_metrics.get('aspect_activity_count', 0),
            human_metrics.get('aspect_accomplishment_count', 0),
            human_metrics.get('aspect_achievement_count', 0)
        ]

        llm_aspects = [
            llm_metrics.get('aspect_state_count', 0),
            llm_metrics.get('aspect_activity_count', 0),
            llm_metrics.get('aspect_accomplishment_count', 0),
            llm_metrics.get('aspect_achievement_count', 0)
        ]

        try:
            if sum(human_aspects) > 0 and sum(llm_aspects) > 0:
                chi2, p_value, dof, expected = chi2_contingency([human_aspects, llm_aspects])
                aspect_score = p_value
            else:
                aspect_score = 1.0 if sum(human_aspects) == sum(llm_aspects) else 0.0
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            aspect_score = 0.0

        overall_score = (
            node_score * 0.3 +
            depth_score * 0.2 +
            reentrancy_score * 0.3 +
            aspect_score * 0.2
        )

        if overall_score > 0.8:
            interpretation = "Excellent"
        elif overall_score > 0.6:
            interpretation = "Good"
        elif overall_score > 0.4:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"

        return {
            "overall_score": round(overall_score, 4),
            "node_ratio": round(node_ratio, 4),
            "node_score": round(node_score, 4),
            "depth_delta": int(depth_delta),
            "depth_score": round(depth_score, 4),
            "reentrancy_ratio": round(reentrancy_ratio, 4),
            "reentrancy_score": round(reentrancy_score, 4),
            "aspect_p_value": round(aspect_score, 4),
            "interpretation": interpretation
        }


def create_complexity_similarity_analyzer() -> ComplexitySimilarityAnalyzer:
    return ComplexitySimilarityAnalyzer()
