import logging
from typing import Dict, Any, List, Set
import penman

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMRValidator:

    VALID_ASPECTS = {'State', 'Activity', 'Performance', 'Habitual', 'Endeavor',
                     'state', 'activity', 'performance', 'habitual', 'endeavor'}

    INVALID_ASPECTS = {'accomplishment', 'achievement', 'Accomplishment', 'Achievement'}

    VALID_MODALS = {'AFF', 'NEG', 'PRT-AFF', 'PRT-NEG', 'Hypothetical', 'Obligative',
                    'Permissive', 'Abilitative', 'Intentional', 'Evidential',
                    'FullAff', 'PartAff', 'FullNeg', 'NeutAff'}

    VALID_TEMPORAL = {':before', ':after', ':overlap', ':contained', ':meets',
                     ':starts-after', ':ends-before', ':includes', ':contains'}

    CORE_ROLES = {':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5'}

    COMMON_EVENT_CONCEPTS = {
        # Motion verbs
        'go', 'come', 'run', 'walk', 'move', 'travel', 'arrive', 'leave', 'enter', 'exit',
        'return', 'fly', 'drive', 'ride', 'swim', 'climb', 'jump', 'fall', 'cross',
        # Communication verbs
        'say', 'tell', 'speak', 'talk', 'ask', 'answer', 'write', 'read', 'communicate',
        'announce', 'declare', 'explain', 'describe', 'report', 'mention', 'suggest',
        # Perception verbs
        'see', 'hear', 'watch', 'listen', 'feel', 'smell', 'taste', 'observe', 'notice',
        'look', 'view', 'perceive', 'detect', 'recognize', 'identify',
        # Cognitive verbs
        'think', 'know', 'believe', 'understand', 'remember', 'forget', 'learn', 'realize',
        'consider', 'assume', 'suppose', 'imagine', 'expect', 'hope', 'decide', 'plan',
        # Basic actions
        'eat', 'drink', 'sleep', 'work', 'play', 'make', 'create', 'build', 'destroy',
        'give', 'take', 'put', 'get', 'bring', 'carry', 'hold', 'use', 'buy', 'sell',
        # State change
        'begin', 'start', 'finish', 'end', 'complete', 'continue', 'stop', 'cease',
        'open', 'close', 'break', 'fix', 'repair', 'change', 'become', 'remain', 'stay',
        # Emotional/volitional
        'want', 'need', 'like', 'love', 'hate', 'try', 'help', 'cause', 'affect',
        'prefer', 'enjoy', 'fear', 'worry', 'surprise', 'satisfy', 'disappoint',
        # Quantitative
        'increase', 'decrease', 'grow', 'shrink', 'expand', 'reduce', 'add', 'remove',
        # Social
        'meet', 'join', 'follow', 'lead', 'support', 'oppose', 'agree', 'disagree',
        'fight', 'attack', 'defend', 'protect', 'serve', 'represent',
        # Romanian common verbs (base forms)
        'merge', 'veni', 'pleca', 'ajunge', 'intra', 'iesi', 'fugi', 'alerga',
        'spune', 'zice', 'vorbi', 'intreba', 'raspunde', 'scrie', 'citi', 'comunica',
        'vedea', 'auzi', 'simti', 'mirosi', 'gusta', 'observa', 'privi',
        'gandi', 'sti', 'crede', 'intelege', 'invata', 'uita', 'aminti', 'decide',
        'manca', 'bea', 'dormi', 'lucra', 'juca', 'face', 'crea', 'construi', 'distruge',
        'da', 'lua', 'pune', 'primi', 'aduce', 'duce', 'tine', 'folosi', 'cumpara', 'vinde',
        'incepe', 'termina', 'continua', 'opri', 'deschide', 'inchide', 'schimba',
        'vrea', 'trebui', 'putea', 'iubi', 'ura', 'incerca', 'ajuta', 'cauza',
        'creste', 'scadea', 'mari', 'reduce', 'adauga', 'elimina',
        'intalni', 'uni', 'urma', 'conduce', 'sprijini', 'sustine'
    }

    UMR_PARTICIPANT_ROLES = {
        ':actor', ':companion', ':instrument', ':force', ':causer', ':cause',
        ':undergoer', ':theme', ':recipient', ':experiencer', ':stimulus', ':material',
        ':goal', ':start', ':source', ':affectee', ':place', ':manner', ':purpose',
        ':reason', ':temporal', ':extent', ':other-role', ':beneficiary',
        ':Actor-of', ':Undergoer-of', ':Stimulus-of', ':Theme-of', ':Experiencer-of',
        ':ARG0-of', ':ARG1-of', ':ARG2-of', ':ARG3-of', ':ARG4-of', ':ARG5-of'
    }

    UMR_NON_PARTICIPANT_ROLES = {
        ':calendar', ':century', ':day', ':dayperiod', ':decade', ':era', ':month',
        ':quarter', ':season', ':time', ':timezone', ':weekday', ':year', ':year2',
        ':mod', ':poss', ':part', ':part-of', ':age', ':group', ':topic', ':medium',
        ':direction', ':path', ':duration', ':frequency', ':location',
        ':name', ':wiki', ':op1', ':op2', ':op3', ':op4', ':op5', ':op6', ':op7', ':op8', ':op9', ':op10',
        ':ord', ':quant', ':range', ':scale', ':unit', ':value',
        ':example', ':polite', ':li', ':condition', ':concession',
        ':aspect', ':mode', ':modstr', ':modpred', ':quot', ':polarity', ':degree',
        ':ref-person', ':ref-number',
        ':same-entity', ':same-event', ':subset-of', ':modal', ':scope',
        ':destination', ':accompanier', ':subevent'
    }

    SENTENCE_LEVEL_ROLES = {
        ':snt1', ':snt2', ':snt3', ':snt4', ':snt5', ':snt6', ':snt7', ':snt8', ':snt9', ':snt10',
        ':snt11', ':snt12', ':snt13', ':snt14', ':snt15', ':snt16', ':snt17', ':snt18', ':snt19', ':snt20'
    }

    ABSTRACT_CONCEPTS = {
        'have-role-91', 'have-mod-91', 'have-91', 'exist-91', 'have-location-91',
        'have-org-role-91', 'have-rel-role-91', 'have-quant-91'
    }

    FORBIDDEN_COPULA = {'be-01', 'be-02', 'be-located-at-91'}

    def __init__(self):
        self.valid_roles = (self.CORE_ROLES | self.UMR_PARTICIPANT_ROLES |
                           self.UMR_NON_PARTICIPANT_ROLES | self.SENTENCE_LEVEL_ROLES)

    def validate_graph(self, penman_str: str, language: str = "english") -> Dict[str, Any]:
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

        self._check_forbidden_copula(graph, errors)
        self._check_aspect_annotation(graph, event_nodes, errors)
        self._check_invalid_aspects(graph, errors)
        self._check_dangling_variables(graph, defined_vars, errors)
        self._check_valid_roles(graph, warnings)
        self._check_modal_values(graph, warnings)
        self._check_temporal_relations(graph, warnings)

        if language == "romanian":
            self._check_romanian_rules(graph, errors, warnings)

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
        if not isinstance(concept, str):
            return False

        if concept.endswith(('-00', '-01', '-02', '-03', '-04', '-05', '-11', '-91', '-21', '-31')):
            return True

        base_concept = concept.split('-')[0] if '-' in concept else concept
        base_concept_lower = base_concept.lower()

        if base_concept_lower in self.COMMON_EVENT_CONCEPTS:
            return True

        return False

    def _has_aspect(self, graph: penman.Graph, node_var: str) -> bool:
        for triple in graph.attributes():
            if triple.source == node_var and triple.role == ':aspect':
                return True
        return False

    def _check_forbidden_copula(self, graph: penman.Graph, errors: List[str]):
        for triple in graph.instances():
            concept = triple.target
            if concept in self.FORBIDDEN_COPULA:
                errors.append(f"Forbidden copula verb '{concept}' found. Use abstract concepts (-91) instead.")

    def _check_aspect_annotation(self, graph: penman.Graph, event_nodes: Set[str], errors: List[str]):
        for node in event_nodes:
            if not self._has_aspect(graph, node):
                concept = self._get_concept(graph, node)
                errors.append(f"Event node '{node}' ({concept}) missing mandatory :aspect annotation")

    def _check_invalid_aspects(self, graph: penman.Graph, errors: List[str]):
        for triple in graph.attributes():
            if triple.role == ':aspect':
                aspect_value = str(triple.target).strip('"').lower()
                if aspect_value in {'accomplishment', 'achievement'}:
                    errors.append(f"Invalid aspect '{triple.target}'. Use 'Performance' instead of accomplishment/achievement.")

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

    def _check_romanian_rules(self, graph: penman.Graph, errors: List[str], warnings: List[str]):
        for triple in graph.instances():
            concept = triple.target
            if concept.endswith(('-01', '-02', '-03', '-04', '-05', '-11', '-21', '-31')):
                if concept not in self.ABSTRACT_CONCEPTS:
                    errors.append(f"Romanian Stage 0: Use native lemma with -00 suffix, not English frame '{concept}'")

        for triple in graph.instances():
            concept = triple.target
            if concept.endswith('-00'):
                node_var = triple.source
                for edge in graph.edges():
                    if edge.source == node_var and edge.role in self.CORE_ROLES:
                        errors.append(f"Romanian Stage 0: Verb '{concept}' uses numbered arg {edge.role}. Use generic roles (:actor, :undergoer, etc.)")
                        break

    def validate_umr_output(self, output: Dict, language: str = "english") -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        all_graph_results = []

        if "sentences" not in output:
            errors.append("Missing 'sentences' key in output")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "quality_score": 0.0
            }

        if not isinstance(output["sentences"], list):
            errors.append("'sentences' must be a list")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "quality_score": 0.0
            }

        for i, sentence in enumerate(output["sentences"]):
            if "id" not in sentence:
                errors.append(f"Sentence {i} missing 'id' field")
            if "graph" not in sentence:
                errors.append(f"Sentence {i} missing 'graph' field")
            else:
                graph_result = self.validate_graph(sentence["graph"], language)
                all_graph_results.append(graph_result)
                if not graph_result["valid"]:
                    for err in graph_result["errors"]:
                        errors.append(f"Sentence {sentence.get('id', i)}: {err}")
                warnings.extend(graph_result.get("warnings", []))

        if "doc_level" not in output:
            warnings.append("Missing 'doc_level' annotations")
        else:
            doc_level = output["doc_level"]
            if "temporal" not in doc_level:
                warnings.append("Missing temporal annotations in doc_level")
            if "modal" not in doc_level:
                warnings.append("Missing modal annotations in doc_level")

        valid = len(errors) == 0
        quality_score = max(0.0, 1.0 - (len(errors) * 0.15 + len(warnings) * 0.03))

        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "quality_score": round(quality_score, 3),
            "sentence_results": all_graph_results
        }

    def repair_graph(self, penman_str: str, language: str = "english") -> str:
        """
        Attempt to automatically repair common UMR graph issues.
        Returns the repaired graph string, or original if repair fails.
        """
        import re

        if not penman_str or not penman_str.strip():
            return penman_str

        repaired = penman_str

        try:
            graph = penman.decode(repaired)
            event_nodes = self._get_event_nodes(graph)
            nodes_missing_aspect = []

            for node in event_nodes:
                if not self._has_aspect(graph, node):
                    concept = self._get_concept(graph, node)
                    nodes_missing_aspect.append((node, concept))

            if nodes_missing_aspect:
                for node_var, concept in nodes_missing_aspect:
                    default_aspect = self._infer_aspect(concept)
                    pattern = rf'\({re.escape(node_var)}\s*/\s*{re.escape(concept)}'
                    replacement = f'({node_var} / {concept} :aspect {default_aspect}'
                    repaired = re.sub(pattern, replacement, repaired)

                logger.info(f"Auto-repaired {len(nodes_missing_aspect)} missing :aspect annotations")

        except Exception as e:
            logger.warning(f"Auto-repair failed: {e}")
            return penman_str

        return repaired

    def _infer_aspect(self, concept: str) -> str:
        """Infer the most likely aspect for a given concept."""
        if not concept:
            return "Performance"

        concept_lower = concept.lower()
        base = concept_lower.split('-')[0] if '-' in concept_lower else concept_lower

        state_verbs = {
            'know', 'believe', 'understand', 'think', 'feel', 'want', 'need', 'like', 'love', 'hate',
            'have', 'own', 'possess', 'exist', 'be', 'seem', 'appear', 'remain', 'stay',
            'sti', 'crede', 'intelege', 'gandi', 'simti', 'vrea', 'avea', 'fi', 'parea', 'ramane'
        }

        activity_verbs = {
            'run', 'walk', 'swim', 'play', 'work', 'talk', 'speak', 'listen', 'watch', 'look',
            'sleep', 'wait', 'read', 'write', 'search', 'study',
            'alerga', 'merge', 'inota', 'juca', 'lucra', 'vorbi', 'asculta', 'privi',
            'dormi', 'astepta', 'citi', 'scrie', 'cauta', 'studia'
        }

        if base in state_verbs or ('have-' in concept_lower and '-91' in concept_lower):
            return "State"
        elif base in activity_verbs:
            return "Activity"
        else:
            return "Performance"

    def repair_umr_output(self, output: Dict, language: str = "english") -> Dict:
        """Attempt to repair all graphs in a UMR output."""
        import copy
        repaired_output = copy.deepcopy(output)

        if "sentences" not in repaired_output:
            return repaired_output

        repairs_made = 0
        for sentence in repaired_output["sentences"]:
            if "graph" in sentence:
                original = sentence["graph"]
                repaired = self.repair_graph(original, language)
                if repaired != original:
                    sentence["graph"] = repaired
                    repairs_made += 1

        if repairs_made > 0:
            logger.info(f"Auto-repaired {repairs_made} sentence graphs")

        return repaired_output
