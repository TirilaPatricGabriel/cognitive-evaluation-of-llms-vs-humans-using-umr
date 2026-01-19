import logging
from typing import Dict, List, Optional, Any
import networkx as nx
import pandas as pd
import penman
from penman.graph import Graph
import smatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMRAnalyzer:

    @staticmethod
    def parse_penman(penman_string: str) -> Optional[Graph]:
        if not penman_string or not penman_string.strip():
            logger.debug("Empty Penman string provided")
            return None

        graph_strings = penman_string.strip().split('\n\n')

        for graph_str in graph_strings:
            graph_str = graph_str.strip()
            if not graph_str:
                continue
            try:
                g = penman.decode(graph_str)
                return g
            except Exception as e:
                logger.debug(f"Failed to parse graph segment: {e}")
                continue

        logger.warning(f"Failed to parse any Penman graph from input")
        return None

    @staticmethod
    def parse_all_penman(penman_string: str) -> List[Graph]:
        """Parse all graphs from a multi-sentence UMR string."""
        if not penman_string or not penman_string.strip():
            return []

        graphs = []
        graph_strings = penman_string.strip().split('\n\n')

        for graph_str in graph_strings:
            graph_str = graph_str.strip()
            if not graph_str:
                continue
            try:
                g = penman.decode(graph_str)
                graphs.append(g)
            except Exception:
                continue

        return graphs

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

    def calculate_smatch(self, human_str: str, llm_str: str, use_official: bool = True) -> Dict[str, float]:
        try:
            if not human_str or not human_str.strip() or not llm_str or not llm_str.strip():
                return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0, "method": "empty"}

            if use_official:
                return self._calculate_smatch_official(human_str, llm_str)
            else:
                return self._calculate_smatch_simple(human_str, llm_str)

        except Exception as e:
            logger.error(f"Smatch calculation failed: {e}")
            return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0, "method": "error"}

    def _calculate_smatch_official(self, human_str: str, llm_str: str) -> Dict[str, float]:
        try:
            best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
                human_str, llm_str,
                verbose=False,
                max_iterations=5
            )

            precision = best_match_num / test_triple_num if test_triple_num > 0 else 0.0
            recall = best_match_num / gold_triple_num if gold_triple_num > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            return {
                "smatch_precision": round(precision, 3),
                "smatch_recall": round(recall, 3),
                "smatch_f1": round(f1, 3),
                "method": "official",
                "matched_triples": int(best_match_num),
                "test_triples": int(test_triple_num),
                "gold_triples": int(gold_triple_num)
            }
        except Exception as e:
            logger.warning(f"Official Smatch failed ({e}), falling back to simple method")
            return self._calculate_smatch_simple(human_str, llm_str)

    def _calculate_smatch_simple(self, human_str: str, llm_str: str) -> Dict[str, float]:
        human_graph = self.parse_penman(human_str)
        llm_graph = self.parse_penman(llm_str)

        if not human_graph or not llm_graph:
            return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0, "method": "simple_failed"}

        human_triples = self._extract_triples(human_graph)
        llm_triples = self._extract_triples(llm_graph)

        if not human_triples and not llm_triples:
            return {"smatch_precision": 1.0, "smatch_recall": 1.0, "smatch_f1": 1.0, "method": "simple"}
        if not llm_triples:
            return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0, "method": "simple"}
        if not human_triples:
            return {"smatch_precision": 0.0, "smatch_recall": 0.0, "smatch_f1": 0.0, "method": "simple"}

        matched = len(human_triples & llm_triples)
        precision = matched / len(llm_triples) if llm_triples else 0.0
        recall = matched / len(human_triples) if human_triples else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "smatch_precision": round(precision, 3),
            "smatch_recall": round(recall, 3),
            "smatch_f1": round(f1, 3),
            "method": "simple_fallback"
        }

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
        all_graphs = self.parse_all_penman(penman_str)

        if not all_graphs:
            return {
                "node_count": 0,
                "graph_depth": 0,
                "reentrancy_count": 0,
                "aspect_state_count": 0,
                "aspect_activity_count": 0,
                "aspect_performance_count": 0,
                "aspect_habitual_count": 0,
                "aspect_endeavor_count": 0,
                "total_aspect_markers": 0
            }

        total_node_count = 0
        max_depth = 0
        total_reentrancy = 0
        aspect_counts = {
            "state": 0,
            "activity": 0,
            "performance": 0,
            "habitual": 0,
            "endeavor": 0,
            "total_events": 0
        }

        for pg in all_graphs:
            G = self.to_networkx(pg)
            total_node_count += G.number_of_nodes()

            try:
                root = pg.top
                if root in G:
                    lengths = nx.shortest_path_length(G, source=root)
                    graph_depth = max(lengths.values()) if lengths else 0
                    max_depth = max(max_depth, graph_depth)
            except Exception:
                pass

            for node in G.nodes():
                if G.nodes[node].get('type') == 'instance':
                    if G.in_degree(node) > 1:
                        total_reentrancy += 1

            for triple in pg.attributes():
                if triple.role == ':aspect':
                    aspect_counts["total_events"] += 1
                    clean_target = str(triple.target).strip('"').lower()
                    if clean_target in aspect_counts:
                        aspect_counts[clean_target] += 1

        return {
            "node_count": total_node_count,
            "graph_depth": max_depth,
            "reentrancy_count": total_reentrancy,
            "aspect_state_count": aspect_counts["state"],
            "aspect_activity_count": aspect_counts["activity"],
            "aspect_performance_count": aspect_counts["performance"],
            "aspect_habitual_count": aspect_counts["habitual"],
            "aspect_endeavor_count": aspect_counts["endeavor"],
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
            "human_aspect_performance": h_metrics.get('aspect_performance_count', 0),
            "human_total_aspects": h_metrics.get('total_aspect_markers', 0),
            "llm_node_count": l_metrics.get('node_count', 0),
            "llm_depth": l_metrics.get('graph_depth', 0),
            "llm_reentrancy": l_metrics.get('reentrancy_count', 0),
            "llm_aspect_state": l_metrics.get('aspect_state_count', 0),
            "llm_aspect_activity": l_metrics.get('aspect_activity_count', 0),
            "llm_aspect_performance": l_metrics.get('aspect_performance_count', 0),
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
                    if instance.endswith(("-00", "-01", "-02", "-03", "-04", "-91")):
                        num_predicates += 1
                    else:
                        num_entities += 1

                in_deg = G.in_degree(node)
                if in_deg > 1:
                    num_reentrancies += 1

                deg = G.degree(node)
                degrees.append(deg)

                if instance in {"and-01", "or-01", "and", "or"}:
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

        for sentence_id, stats in sentence_umr_stats.items():
            for key, value in stats.items():
                if isinstance(value, float):
                    stats[key] = round(value, 2)

        return sentence_umr_stats


    def analyze_nodes(self, umr_data: list) -> pd.DataFrame:
        rows = []

        for item in umr_data:
            sentence_id = item.get("sentence_id")
            penman_str = item.get("umr_graph", "")
            if not penman_str:
                continue

            pg = self.parse_penman(penman_str)
            if not pg:
                continue

            G = self.to_networkx(pg)
            root = pg.top

            try:
                depths = nx.shortest_path_length(G, source=root)
            except Exception:
                depths = {}

            node_to_words = {}
            for node in G.nodes():
                if sentence_id == 0:
                    print(f"Node: {node}")
                data = G.nodes[node]
                node_type = data.get("type")
                instance = data.get("instance") if node_type == "instance" else None

                words = []

                if instance == "name" and node_type == "instance":
                    for succ in G.successors(node):
                        succ_data = G.nodes[succ]
                        if succ_data.get("type") == "attribute" and ":op" in succ:
                            val = succ_data.get("value", "").strip('"')
                            if val:
                                words.append(val)

                elif instance == "date-entity":
                    for succ in G.successors(node):
                        succ_data = G.nodes[succ]
                        if succ_data.get("type") == "attribute" and ":year" in succ:
                            val = succ_data.get("value")
                            if val:
                                words.append(str(val))

                elif instance:
                    words.append(instance)

                if sentence_id == 0:
                    print(f"Words for node {node}: {words}")
                node_to_words[node] = words

            for node in G.nodes():
                data = G.nodes[node]
                node_type = data.get("type")
                instance = data.get("instance") if node_type == "instance" else None

                word_list = node_to_words.get(node, [])
                for word_index, word in enumerate(word_list):
                    row = {
                        "sentence_id": sentence_id,
                        "node_id": node,
                        "node_type": node_type,
                        "instance": instance,
                        "word_index": word_index,
                        "word": word,
                        "depth": depths.get(node, 0),
                        "degree": G.degree(node),
                        "in_degree": G.in_degree(node),
                        "out_degree": G.out_degree(node)
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(by=["sentence_id", "word_index"]).reset_index(drop=True)
        return df


def create_umr_analyzer() -> UMRAnalyzer:
    return UMRAnalyzer()
