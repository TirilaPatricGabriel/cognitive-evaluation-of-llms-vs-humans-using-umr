import logging
from typing import Dict, List, Optional, Any
import networkx as nx
import penman
from penman.graph import Graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMRAnalyzer:

    @staticmethod
    def parse_penman(penman_string: str) -> Optional[Graph]:
        try:
            g = penman.decode(penman_string)
            return g
        except Exception as e:
            logger.warning(f"Failed to parse Penman string: {e}")
            return None

    @staticmethod
    def to_networkx(penman_graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()

        for triple in penman_graph.instances():
            source, role, target = triple
            G.add_node(source, instance=target, type='instance')

        for triple in penman_graph.edges():
            source, role, target = triple
            G.add_edge(source, target, label=role)

        for triple in penman_graph.attributes():
            source, role, target = triple
            attr_node_id = f"{source}_{role}_{target}"
            G.add_node(attr_node_id, type='attribute', value=target)
            G.add_edge(source, attr_node_id, label=role)

        return G

    def calculate_smatch(self, human_str: str, llm_str: str) -> Dict[str, float]:
        try:
            human_graph = self.parse_penman(human_str)
            llm_graph = self.parse_penman(llm_str)

            if not human_graph or not llm_graph:
                return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0}

            human_triples = self._extract_triples(human_graph)
            llm_triples = self._extract_triples(llm_graph)

            if not human_triples and not llm_triples:
                return {"smatch_precision": 1.0, "smatch_recall": 1.0, "smatch_f1": 1.0}
            if not llm_triples:
                return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0}
            if not human_triples:
                return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0}

            matched = len(human_triples & llm_triples)
            precision = matched / len(llm_triples) if llm_triples else 0.0
            recall = matched / len(human_triples) if human_triples else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            return {
                "smatch_precision": round(precision, 3),
                "smatch_recall": round(recall, 3),
                "smatch_f1": round(f1, 3)
            }
        except Exception as e:
            logger.error(f"Smatch calculation failed: {e}")
            return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0}

    def _extract_triples(self, graph: Graph) -> set:
        triples = set()

        for triple in graph.instances():
            triples.add(('instance', triple.source, triple.target))

        for triple in graph.edges():
            triples.add(('edge', triple.source, triple.role, triple.target))

        for triple in graph.attributes():
            triples.add(('attr', triple.source, triple.role, triple.target))

        return triples

    def analyze_complexity(self, penman_str: str) -> Dict[str, Any]:
        pg = self.parse_penman(penman_str)
        if not pg:
            return {
                "node_count": 0,
                "graph_depth": 0,
                "reentrancy_count": 0,
                "aspect_state_count": 0,
                "aspect_activity_count": 0,
                "aspect_accomplishment_count": 0,
                "aspect_achievement_count": 0,
                "total_aspect_markers": 0
            }

        G = self.to_networkx(pg)

        node_count = G.number_of_nodes()

        try:
            root = pg.top
            if root in G:
                lengths = nx.shortest_path_length(G, source=root)
                max_depth = max(lengths.values()) if lengths else 0
            else:
                max_depth = 0
        except Exception:
            max_depth = 0

        reentrancy_count = 0
        for node in G.nodes():
            if G.nodes[node].get('type') == 'instance':
                if G.in_degree(node) > 1:
                    reentrancy_count += 1

        aspect_counts = {
            "state": 0,
            "activity": 0,
            "accomplishment": 0,
            "achievement": 0,
            "habitual": 0,
            "total_events": 0
        }

        for triple in pg.attributes():
            if triple.role == ':aspect':
                aspect_counts["total_events"] += 1
                clean_target = str(triple.target).strip('"').lower()
                if clean_target in aspect_counts:
                    aspect_counts[clean_target] += 1

        return {
            "node_count": node_count,
            "graph_depth": max_depth,
            "reentrancy_count": reentrancy_count,
            "aspect_state_count": aspect_counts["state"],
            "aspect_activity_count": aspect_counts["activity"],
            "aspect_accomplishment_count": aspect_counts["accomplishment"],
            "aspect_achievement_count": aspect_counts["achievement"],
            "total_aspect_markers": aspect_counts["total_events"]
        }

    def compare_pair(self, human_umr: str, llm_umr: str, text_id: str = "") -> Dict[str, Any]:
        h_metrics = self.analyze_complexity(human_umr)
        l_metrics = self.analyze_complexity(llm_umr)
        similarity = self.calculate_smatch(human_umr, llm_umr)

        return {
            "text_id": text_id,
            "smatch_precision": similarity['smatch_precision'],
            "smatch_recall": similarity['smatch_recall'],
            "smatch_f1": similarity['smatch_f1'],
            "human_node_count": h_metrics.get('node_count', 0),
            "human_depth": h_metrics.get('graph_depth', 0),
            "human_reentrancy": h_metrics.get('reentrancy_count', 0),
            "human_aspect_state": h_metrics.get('aspect_state_count', 0),
            "human_aspect_activity": h_metrics.get('aspect_activity_count', 0),
            "human_total_aspects": h_metrics.get('total_aspect_markers', 0),
            "llm_node_count": l_metrics.get('node_count', 0),
            "llm_depth": l_metrics.get('graph_depth', 0),
            "llm_reentrancy": l_metrics.get('reentrancy_count', 0),
            "llm_aspect_state": l_metrics.get('aspect_state_count', 0),
            "llm_aspect_activity": l_metrics.get('aspect_activity_count', 0),
            "llm_total_aspects": l_metrics.get('total_aspect_markers', 0),
            "delta_nodes": l_metrics.get('node_count', 0) - h_metrics.get('node_count', 0),
            "delta_depth": l_metrics.get('graph_depth', 0) - h_metrics.get('graph_depth', 0),
            "delta_reentrancy": l_metrics.get('reentrancy_count', 0) - h_metrics.get('reentrancy_count', 0)
        }

    def analyze_corpus(self, human_graphs: List[Dict], llm_graphs: List[Dict]) -> List[Dict]:
        results = []

        for h_data, l_data in zip(human_graphs, llm_graphs):
            h_umr = h_data.get('umr_graph', '')
            l_umr = l_data.get('umr_graph', '')

            text_id = h_data.get('filename') or h_data.get('subcategory', 'unknown')

            comparison = self.compare_pair(h_umr, l_umr, text_id)
            comparison['language'] = h_data.get('language', '')
            comparison['subcategory'] = h_data.get('subcategory', '')

            results.append(comparison)

        return results

    def calculate_aggregate_stats(self, comparisons: List[Dict]) -> Dict[str, Any]:
        if not comparisons:
            return {}

        total = len(comparisons)

        avg_smatch_f1 = sum(c['smatch_f1'] for c in comparisons) / total
        avg_smatch_precision = sum(c['smatch_precision'] for c in comparisons) / total
        avg_smatch_recall = sum(c['smatch_recall'] for c in comparisons) / total

        high_fidelity = sum(1 for c in comparisons if c['smatch_f1'] > 0.80)
        low_fidelity = sum(1 for c in comparisons if c['smatch_f1'] < 0.60)

        avg_delta_nodes = sum(c['delta_nodes'] for c in comparisons) / total
        avg_delta_depth = sum(c['delta_depth'] for c in comparisons) / total
        avg_delta_reentrancy = sum(c['delta_reentrancy'] for c in comparisons) / total

        return {
            "total_pairs": total,
            "avg_smatch_f1": round(avg_smatch_f1, 3),
            "avg_smatch_precision": round(avg_smatch_precision, 3),
            "avg_smatch_recall": round(avg_smatch_recall, 3),
            "high_fidelity_count": high_fidelity,
            "low_fidelity_count": low_fidelity,
            "avg_delta_nodes": round(avg_delta_nodes, 2),
            "avg_delta_depth": round(avg_delta_depth, 2),
            "avg_delta_reentrancy": round(avg_delta_reentrancy, 2),
            "interpretation": {
                "semantic_fidelity": "Excellent" if avg_smatch_f1 > 0.80 else "Moderate" if avg_smatch_f1 > 0.60 else "Significant Divergence",
                "complexity_trend": "LLM adds complexity" if avg_delta_nodes > 10 else "LLM simplifies" if avg_delta_nodes < -10 else "Similar complexity",
                "coherence_trend": "More connected (LLM)" if avg_delta_reentrancy > 0 else "Less connected (LLM)" if avg_delta_reentrancy < 0 else "Similar coherence"
            }
        }
        
        
    def analyze_sentences(self, umr_data) -> Dict[int, Any]:
        """
        Compute sentence-level UMR graph statistics for comparison with sentence-level eye-tracking measures.
        """

        logging.getLogger("penman").setLevel(logging.WARNING)
        
        sentence_umr_stats = {}

        for item in umr_data:
            sentence_id = item.get("sentence_id", "unknown")
            penman_str = item.get("umr_graph", "")

            pg = self.parse_penman(penman_str)
            if not pg:
                print(f"Failed to parse UMR graph for sentence {sentence_id}")
                continue

            G = self.to_networkx(pg)
            root = pg.top

            try:
                depths = nx.shortest_path_length(G, source=root)
                depth_values = list(depths.values())
            except Exception:
                depth_values = []

            num_nodes = 0
            num_predicates = 0
            num_entities = 0
            num_reentrancies = 0
            degrees = []

            num_coordination = 0
            num_temporal_quantities = 0

            for node in G.nodes():
                data = G.nodes[node]

                if data.get("type") != "instance":
                    continue

                num_nodes += 1
                instance = data.get("instance") 
                
                if isinstance(instance, str):
                    if instance.endswith(("-01", "-02", "-03", "-04")):
                        num_predicates += 1
                    else:
                        num_entities += 1

                in_deg = G.in_degree(node)
                if in_deg > 1:
                    num_reentrancies += 1

                deg = G.degree(node)
                degrees.append(deg)

                if instance in {"and-01", "or-01"}:
                    num_coordination += 1

                if instance == "temporal-quantity":
                    num_temporal_quantities += 1

            num_edges = G.number_of_edges()

            sentence_umr_stats[sentence_id] = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "max_depth": max(depth_values) if depth_values else 0,
                "avg_depth": sum(depth_values) / len(depth_values) if depth_values else 0.0,
                "num_predicates": num_predicates,
                "num_entities": num_entities,
                "predicate_entity_ratio": (num_predicates / num_entities if num_entities > 0 else 0.0),
                "num_reentrancies": num_reentrancies,
                "avg_degree": (sum(degrees) / len(degrees) if degrees else 0.0),
                "max_degree": max(degrees) if degrees else 0,
                "num_coordination": num_coordination,
                "num_temporal_quantities": num_temporal_quantities
            }

        # Limit to 2 decimal places
        for sentence_id, stats in sentence_umr_stats.items():
            for key, value in stats.items():
                if isinstance(value, float):
                    stats[key] = round(value, 2)
                    
        return sentence_umr_stats


def create_umr_analyzer() -> UMRAnalyzer:
    return UMRAnalyzer()
