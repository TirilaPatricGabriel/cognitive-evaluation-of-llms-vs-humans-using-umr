import logging
from typing import Dict, Any, List, Set
import penman

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMRValidator:

    VALID_ASPECTS = {'state', 'activity', 'accomplishment', 'achievement', 'habitual',
                     'endeavor', 'performance', 'process'}  # Added accomplishment/achievement per UMR 1.0

    VALID_MODALS = {'AFF', 'NEG', 'PRT-AFF', 'PRT-NEG', 'Hypothetical', 'Obligative',
                    'Permissive', 'Abilitative', 'Intentional', 'Evidential'}

    VALID_TEMPORAL = {':before', ':after', ':overlap', ':contained', ':meets',
                     ':starts-after', ':ends-before', ':includes'}

    CORE_ROLES = {':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5'}

    NON_CORE_ROLES = {':time', ':location', ':manner', ':topic', ':medium', ':purpose',
                     ':cause', ':consist-of', ':part-of', ':poss', ':mod', ':polarity',
                     ':aspect', ':quant', ':degree', ':domain', ':op1', ':op2', ':op3',
                     ':op4', ':op5', ':op6', ':name', ':condition', ':concession', ':temporal', ':modal',
                     ':same-entity', ':scope', ':ARG0-of', ':ARG1-of', ':ARG2-of', ':ARG3-of', ':ARG4-of',
                     ':snt1', ':snt2', ':snt3', ':snt4', ':snt5', ':snt6', ':snt7', ':snt8', ':snt9', ':snt10',
                     ':snt11', ':snt12', ':snt13', ':snt14', ':snt15', ':snt16', ':snt17', ':snt18', ':snt19', ':snt20',
                     ':source', ':destination', ':path', ':frequency', ':duration', ':extent', ':value',
                     ':instrument', ':beneficiary', ':accompanier', ':direction', ':subevent'}

    def __init__(self):
        self.valid_roles = self.CORE_ROLES | self.NON_CORE_ROLES

    def validate_graph(self, penman_str: str) -> Dict[str, Any]:
        if not penman_str or not penman_str.strip():
            return {
                "valid": False,
                "error": "Empty UMR graph string",
                "errors": ["Empty input"],
                "warnings": [],
                "quality_score": 0.0
            }

        try:
            graph = penman.decode(penman_str)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Parse error: {str(e)}",
                "errors": [f"Penman parse failed: {str(e)}"],
                "warnings": [],
                "quality_score": 0.0
            }

        errors: List[str] = []
        warnings: List[str] = []

        defined_vars = self._get_defined_variables(graph)
        event_nodes = self._get_event_nodes(graph)

        self._check_aspect_annotation(graph, event_nodes, warnings)
        self._check_dangling_variables(graph, defined_vars, errors)
        self._check_valid_roles(graph, warnings)
        self._check_modal_values(graph, warnings)
        self._check_temporal_relations(graph, warnings)

        valid = len(errors) == 0
        quality_score = max(0.0, 1.0 - (len(errors) * 0.2 + len(warnings) * 0.05))

        return {
            "valid": valid,
            "error": errors[0] if errors else None,
            "errors": errors,
            "warnings": warnings,
            "quality_score": round(quality_score, 3),
            "stats": {
                "total_nodes": len(defined_vars),
                "event_nodes": len(event_nodes),
                "has_aspect_annotations": sum(1 for n in event_nodes if self._has_aspect(graph, n)),
                "error_count": len(errors),
                "warning_count": len(warnings)
            }
        }

    def _get_defined_variables(self, graph: penman.Graph) -> Set[str]:
        return {triple.source for triple in graph.instances()}

    def _get_event_nodes(self, graph: penman.Graph) -> Set[str]:
        event_nodes = set()
        for triple in graph.instances():
            concept = triple.target
            if self._is_event_concept(concept):
                event_nodes.add(triple.source)
        return event_nodes

    def _is_event_concept(self, concept: str) -> bool:
        """
        Determine if a concept is an event (requires :aspect annotation).

        Events are:
        1. PropBank-style frames: ending in -01, -02, -03, -04, -05, -11, -91
        2. Common event predicates (verbs that describe actions/states)
        """
        if not isinstance(concept, str):
            return False

        # PropBank-style frames (most reliable indicator)
        if concept.endswith(('-01', '-02', '-03', '-04', '-05', '-11', '-91', '-21', '-31')):
            return True

        # Common event/state predicates
        # These are verbs that typically require aspect in UMR
        common_events = {
            # Motion verbs
            'go', 'come', 'run', 'walk', 'move', 'travel', 'arrive', 'leave', 'enter', 'exit',
            # Communication verbs
            'say', 'tell', 'speak', 'talk', 'ask', 'answer', 'write', 'read', 'communicate',
            # Perception verbs
            'see', 'hear', 'watch', 'listen', 'feel', 'smell', 'taste', 'observe', 'notice',
            # Cognition verbs
            'think', 'know', 'believe', 'understand', 'remember', 'forget', 'learn', 'realize',
            # Action verbs
            'eat', 'drink', 'sleep', 'work', 'play', 'make', 'create', 'build', 'destroy',
            'give', 'take', 'put', 'get', 'bring', 'carry', 'hold', 'use',
            # Change/completion verbs
            'begin', 'start', 'finish', 'end', 'complete', 'continue', 'stop', 'cease',
            # Other common events
            'want', 'need', 'like', 'love', 'hate', 'try', 'help', 'cause', 'affect',
            'increase', 'decrease', 'change', 'become', 'remain', 'stay'
        }

        # Check base concept (before hyphen)
        base_concept = concept.split('-')[0] if '-' in concept else concept
        if base_concept in common_events:
            return True

        return False

    def _has_aspect(self, graph: penman.Graph, node_var: str) -> bool:
        for triple in graph.attributes():
            if triple.source == node_var and triple.role == ':aspect':
                return True
        return False

    def _check_aspect_annotation(self, graph: penman.Graph, event_nodes: Set[str], warnings: List[str]):
        for node in event_nodes:
            if not self._has_aspect(graph, node):
                concept = self._get_concept(graph, node)
                warnings.append(f"Event node '{node}' ({concept}) missing :aspect annotation")

    def _get_concept(self, graph: penman.Graph, var: str) -> str:
        for triple in graph.instances():
            if triple.source == var:
                return triple.target
        return "unknown"

    def _check_dangling_variables(self, graph: penman.Graph, defined_vars: Set[str], errors: List[str]):
        for triple in graph.edges():
            if triple.target not in defined_vars and not self._is_literal(triple.target):
                errors.append(f"Undefined variable reference: {triple.target} in {triple.source} {triple.role}")

    def _is_literal(self, value: Any) -> bool:
        if isinstance(value, (int, float, bool)):
            return True
        if isinstance(value, str):
            if value.startswith('"') and value.endswith('"'):
                return True
            if value in ['+', '-', 'interrogative', 'expressive', 'imperative']:
                return True
        return False

    def _check_valid_roles(self, graph: penman.Graph, warnings: List[str]):
        for triple in list(graph.edges()) + list(graph.attributes()):
            if triple.role not in self.valid_roles:
                warnings.append(f"Non-standard role: {triple.role} in {triple.source}")

    def _check_modal_values(self, graph: penman.Graph, warnings: List[str]):
        for triple in graph.attributes():
            if triple.role == ':modal':
                if triple.target not in self.VALID_MODALS:
                    warnings.append(f"Invalid modal value: {triple.target} (expected one of {self.VALID_MODALS})")

    def _check_temporal_relations(self, graph: penman.Graph, warnings: List[str]):
        for triple in graph.edges():
            if triple.role in self.VALID_TEMPORAL:
                if triple.target not in self._get_defined_variables(graph):
                    warnings.append(f"Temporal relation {triple.role} points to undefined variable: {triple.target}")
