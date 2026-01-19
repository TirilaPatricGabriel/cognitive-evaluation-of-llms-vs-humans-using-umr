import re
from typing import Dict, List, Tuple, Optional


class UMRVisualizer:

    @staticmethod
    def parse_penman_structure(umr_graph: str) -> Dict:
        umr_graph = umr_graph.strip()

        stats = {
            'total_chars': len(umr_graph),
            'depth': UMRVisualizer._calculate_depth(umr_graph),
            'num_concepts': len(re.findall(r'/\s*[\w\-]+', umr_graph)),
            'num_roles': len(re.findall(r':\w+', umr_graph)),
            'num_variables': len(re.findall(r'\(\s*(\w+)\s*/', umr_graph)),
            'has_aspect': ':aspect' in umr_graph,
            'has_temporal': ':time' in umr_graph or ':temporal' in umr_graph,
            'has_modality': ':mode' in umr_graph or ':modal' in umr_graph,
        }

        aspects = re.findall(r':aspect\s+([\w\-]+)', umr_graph)
        roles = re.findall(r':(ARG\d+|op\d+|time|mod|aspect|polarity|actor|undergoer|theme|recipient|experiencer)', umr_graph)
        concepts = re.findall(r'/\s*([\w\-]+)', umr_graph)

        return {
            'stats': stats,
            'aspects': list(set(aspects)),
            'roles': list(set(roles)),
            'concepts': concepts[:10],
            'main_concept': concepts[0] if concepts else None
        }

    @staticmethod
    def _calculate_depth(umr_graph: str) -> int:
        max_depth = 0
        current_depth = 0

        for char in umr_graph:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        return max_depth

    @staticmethod
    def format_for_display(umr_graph: str) -> str:
        if not umr_graph:
            return ""

        lines = []
        indent_level = 0
        current_line = ""

        i = 0
        while i < len(umr_graph):
            char = umr_graph[i]

            if char == '(':
                if current_line.strip():
                    lines.append("  " * indent_level + current_line.strip())
                    current_line = ""
                lines.append("  " * indent_level + "(")
                indent_level += 1
            elif char == ')':
                if current_line.strip():
                    lines.append("  " * indent_level + current_line.strip())
                    current_line = ""
                indent_level -= 1
                lines.append("  " * indent_level + ")")
            elif char == '\n':
                if current_line.strip():
                    lines.append("  " * indent_level + current_line.strip())
                    current_line = ""
            else:
                current_line += char

            i += 1

        if current_line.strip():
            lines.append("  " * indent_level + current_line.strip())

        cleaned_lines = [line for line in lines if line.strip()]

        return "\n".join(cleaned_lines)

    @staticmethod
    def compare_graphs(graph1: str, graph2: str) -> Dict:
        struct1 = UMRVisualizer.parse_penman_structure(graph1)
        struct2 = UMRVisualizer.parse_penman_structure(graph2)

        comparison = {
            'depth_diff': abs(struct1['stats']['depth'] - struct2['stats']['depth']),
            'concept_count_diff': abs(struct1['stats']['num_concepts'] - struct2['stats']['num_concepts']),
            'role_count_diff': abs(struct1['stats']['num_roles'] - struct2['stats']['num_roles']),
            'shared_aspects': list(set(struct1['aspects']) & set(struct2['aspects'])),
            'shared_roles': list(set(struct1['roles']) & set(struct2['roles'])),
            'both_have_aspect': struct1['stats']['has_aspect'] and struct2['stats']['has_aspect'],
            'both_have_temporal': struct1['stats']['has_temporal'] and struct2['stats']['has_temporal'],
        }

        return comparison

    @staticmethod
    def create_visualization_summary(parsed_result: Dict) -> Dict:
        umr_graph = ""

        if 'sentences' in parsed_result:
            graphs = []
            for sent in parsed_result.get('sentences', []):
                if 'graph' in sent:
                    graphs.append(sent['graph'])
            umr_graph = "\n\n".join(graphs)
        elif 'umr_graph' in parsed_result:
            umr_graph = parsed_result.get('umr_graph', '')

        if not umr_graph:
            return {
                'error': 'No UMR graph available',
                'success': False
            }

        structure = UMRVisualizer.parse_penman_structure(umr_graph)
        formatted = UMRVisualizer.format_for_display(umr_graph)

        return {
            'success': True,
            'formatted_graph': formatted,
            'structure': structure,
            'original_text': parsed_result.get('original_text', ''),
            'language': parsed_result.get('language', ''),
            'parsing_success': parsed_result.get('success', False)
        }


def visualize_umr_graph(umr_graph: str) -> str:
    return UMRVisualizer.format_for_display(umr_graph)


def analyze_umr_structure(umr_graph: str) -> Dict:
    return UMRVisualizer.parse_penman_structure(umr_graph)
