import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))

from app.utils.bertscore_calculator import create_bertscore_calculator
from app.utils.soft_smatch import create_soft_smatch_calculator
from app.utils.concept_overlap import create_concept_overlap_calculator
from app.utils.complexity_similarity import create_complexity_similarity_analyzer
from app.services.umr_analyzer import create_umr_analyzer

print("=" * 80)
print("COMPREHENSIVE UMR EVALUATION FRAMEWORK - TEST SUITE")
print("=" * 80)
print("Using REAL data from: Lit_Alchemist_ro.txt")
print("=" * 80)

import json
with open('sample_pair.json', 'r', encoding='utf-8') as f:
    sample = json.load(f)

human_text = sample['human']['text'][:500]
llm_text = sample['llm']['text'][:500]
human_umr = sample['human']['umr_graph']
llm_umr = sample['llm']['umr_graph']

print(f"\nHuman text preview: {human_text[:100]}...")
print(f"LLM text preview: {llm_text[:100]}...")
print(f"Human UMR length: {len(human_umr)} chars")
print(f"LLM UMR length: {len(llm_umr)} chars")
print("=" * 80)

print("\n1. Testing BERTScore Calculator...")
bertscore_calc = create_bertscore_calculator()
bertscore_result = bertscore_calc.calculate_single(human_text, llm_text, "romanian")
print(f"   BERTScore F1: {bertscore_result['f1']:.4f}")
print(f"   ✓ BERTScore calculator working")

print("\n2. Testing Concept Overlap Calculator...")
concept_overlap_calc = create_concept_overlap_calculator()
concept_result = concept_overlap_calc.calculate(human_umr, llm_umr)
print(f"   Concept Overlap F1: {concept_result['f1']:.4f}")
print(f"   Shared concepts: {concept_result['shared_count']}/{concept_result['human_count']}")
print(f"   ✓ Concept Overlap calculator working")

print("\n3. Testing Soft-Smatch Calculator...")
soft_smatch_calc = create_soft_smatch_calculator()
soft_result = soft_smatch_calc.calculate(human_umr, llm_umr, threshold=0.75)
print(f"   Soft-Smatch F1: {soft_result['f1']:.4f}")
print(f"   Matched pairs: {soft_result['matched_pairs']}")
print(f"   ✓ Soft-Smatch calculator working")

print("\n4. Testing Standard Smatch (for comparison)...")
umr_analyzer = create_umr_analyzer()
smatch_result = umr_analyzer.calculate_smatch(human_umr, llm_umr)
print(f"   Standard Smatch F1: {smatch_result['smatch_f1']:.4f}")
print(f"   ✓ Standard Smatch working")

print("\n5. Testing Complexity Similarity Analyzer...")
complexity_analyzer = create_complexity_similarity_analyzer()
human_complexity = umr_analyzer.analyze_complexity(human_umr)
llm_complexity = umr_analyzer.analyze_complexity(llm_umr)
complexity_result = complexity_analyzer.calculate(human_complexity, llm_complexity)
print(f"   Complexity Score: {complexity_result['overall_score']:.4f}")
print(f"   Node Ratio: {complexity_result['node_ratio']:.4f}")
print(f"   Interpretation: {complexity_result['interpretation']}")
print(f"   ✓ Complexity Similarity analyzer working")

print("\n6. Testing Comprehensive Evaluator...")
from app.services.comprehensive_umr_evaluator import create_comprehensive_evaluator
evaluator = create_comprehensive_evaluator()

result = evaluator.evaluate_pair(
    human_text=human_text,
    llm_text=llm_text,
    human_umr=human_umr,
    llm_umr=llm_umr,
    text_id="test_sample",
    language="romanian"
)

print(f"   Overall Score: {result['overall_assessment']['overall_score']:.4f}")
print(f"   Interpretation: {result['overall_assessment']['interpretation']}")
print(f"   ✓ Comprehensive Evaluator working")

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nYou can now run the comprehensive evaluation on your full dataset using:")
print("  POST http://localhost:8000/analyze-umr-semantics-comprehensive")
print("\nThis will evaluate all text pairs using:")
print("  - BERTScore (text-level semantic similarity)")
print("  - Soft-Smatch (embedding-based concept matching)")
print("  - Concept Overlap (simple concept set comparison)")
print("  - Standard Smatch (graph isomorphism baseline)")
print("  - Complexity Similarity (structural pattern matching)")
print("=" * 80)
