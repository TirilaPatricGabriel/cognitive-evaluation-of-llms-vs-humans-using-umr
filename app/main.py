import pandas as pd
from fastapi import FastAPI
from app.loaders import load_multipleye_data, load_zuco2_data
from app.services import call_gemini, create_umr_parser, create_umr_analyzer, UMREyeTrackingComparator
from app.utils.load_yaml_prompts import load_prompt
from app.utils.umr_visualizer import UMRVisualizer
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"Global exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"message": str(exc), "traceback": traceback.format_exc()},
    )

@app.get("/")
def root():
    return {"status": "ok", "message": "UMR Cognitive Evaluation API"}

@app.post("/healthcheck")
def healthcheck():
    return {"message": "healthy"}


@app.get("/load-data/{dataset_name}")
def load_data(dataset_name: str):
    if dataset_name == "multipleye":
        df_ro, df_en = load_multipleye_data()

        return {
            "romanian": {
                "count": len(df_ro),
                "sample": df_ro.head(1).to_dict(orient="records")
            },
            "english": {
                "count": len(df_en),
                "sample": df_en.head(1).to_dict(orient="records")
            }
        }
    elif dataset_name == "zuco2":
        df_zuco = load_zuco2_data()
        return {
            "zuco2": {
                "count": len(df_zuco),
                "sample": df_zuco.head(1).to_dict(orient="records")
            }
        }


@app.post("/reverse-engineer")
def reverse_engineer(limit: int = None):
    print("Starting reverse engineering")
    df_ro, df_en = load_multipleye_data()

    if limit:
        print(f"Limiting to {limit} texts per language")
        df_ro = df_ro.head(limit)
        df_en = df_en.head(limit)

    prompt_template = load_prompt("app/prompts/reverse_engineer.yaml")

    tasks = []
    for _, row in df_ro.iterrows():
        text = row['text']
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
        
        tasks.append({
            "language": "romanian",
            "subcategory": row['subcategory'],
            "filename": row['filename'],
            "original_text": text,
            "prompt": prompt_template.format(
                text=text,
                word_count=word_count,
                sentence_count=sentence_count,
                avg_len=avg_len
            )
        })

    for _, row in df_en.iterrows():
        text = row.get('text', '')
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0

        tasks.append({
            "language": "english",
            "subcategory": row['subcategory'],
            "original_text": text,
            "prompt": prompt_template.format(
                text=text,
                word_count=word_count,
                sentence_count=sentence_count,
                avg_len=avg_len
            )
        })

    print(f"Processing {len(tasks)} texts concurrently")

    def process_task(task):
        reversed_prompt = call_gemini(task["prompt"])
        result = {
            "language": task["language"],
            "subcategory": task["subcategory"],
            "reversed_prompt": reversed_prompt
        }
        if "filename" in task:
            result["filename"] = task["filename"]
        if "original_text" in task:
            result["original_text"] = task["original_text"]
        return result

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_task, task): i for i, task in enumerate(tasks)}
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{idx}/{len(tasks)}] Done")

    save_dir = Path("app/saved_outputs/prompts")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = save_dir / f"reversed_prompts_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_file}")
    print("Reverse engineering complete")

    return {"prompts": results, "count": len(results), "saved_to": str(output_file)}


@app.post("/generate-texts")
def generate_texts(input_file: str = None):
    print("Starting text generation")

    if not input_file:
        prompts_dir = Path("app/saved_outputs/prompts")
        prompt_files = sorted(prompts_dir.glob("reversed_prompts_*.json"))
        if not prompt_files:
            return {"error": "No reversed prompts files found"}
        input_file = str(prompt_files[-1])

    print(f"Loading prompts from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        items = json.load(f)

    prompt_template = load_prompt("app/prompts/generate_text.yaml")
    total = len(items)

    print(f"Generating {total} texts concurrently")

    def process_item(item):
        original_text = item.get("original_text", "")
        word_count = len(original_text.split()) if original_text else 500
        sentence_count = len([s for s in original_text.split('.') if s.strip()]) if original_text else 25
        prompt = prompt_template.format(
            prompt=item["reversed_prompt"],
            word_count=word_count,
            sentence_count=sentence_count
        )
        generated_text = call_gemini(prompt, temperature=0.7)
        return {
            "language": item["language"],
            "subcategory": item["subcategory"],
            "reversed_prompt": item["reversed_prompt"],
            "generated_text": generated_text
        }

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_item, item): i for i, item in enumerate(items)}
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{idx}/{total}] Done")

    save_dir = Path("app/saved_outputs/generated_texts")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = save_dir / f"generated_texts_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_file}")
    print("Text generation complete")

    return {"generated_texts": results, "count": len(results), "saved_to": str(output_file)}


@app.post("/full-pipeline")
def full_pipeline(limit: int = None):
    """
    Complete pipeline: reverse engineer -> generate texts -> create UMR graphs -> visualize.
    """
    print("=" * 80)
    print("STARTING FULL PIPELINE WITH UMR ANALYSIS")
    print("=" * 80)

    df_ro, df_en = load_multipleye_data()

    if limit:
        print(f"Limiting to {limit} texts per language")
        df_ro = df_ro.head(limit)
        df_en = df_en.head(limit)

    reverse_template = load_prompt("app/prompts/reverse_engineer.yaml")
    generate_template = load_prompt("app/prompts/generate_text.yaml")

    tasks = []
    print("Processing Romanian texts...")
    for _, row in df_ro.iterrows():
        text = row['text']
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
        
        tasks.append({
            "language": "romanian",
            "subcategory": row['subcategory'],
            "filename": row['filename'],
            "original_text": text,
            "prompt": reverse_template.format(
                text=text, 
                word_count=word_count,
                sentence_count=sentence_count,
                avg_len=avg_len
            )
        })

    print("Processing English texts...")
    for _, row in df_en.iterrows():
        text = row.get('text', '')
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
        
        tasks.append({
            "language": "english",
            "subcategory": row['subcategory'],
            "original_text": text,
            "prompt": reverse_template.format(
                text=text, 
                word_count=word_count,
                sentence_count=sentence_count,
                avg_len=avg_len
            )
        })


    print(f"\nSTEP 1/8: Reverse engineering {len(tasks)} human texts")
    print("-" * 80)

    def reverse_task(task):
        reversed_prompt = call_gemini(task["prompt"])
        result = {
            "language": task["language"],
            "subcategory": task["subcategory"],
            "reversed_prompt": reversed_prompt
        }
        if "filename" in task:
            result["filename"] = task["filename"]
        if "original_text" in task:
            result["original_text"] = task["original_text"]
        return result

    reversed_prompts = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(reverse_task, task): i for i, task in enumerate(tasks)}
        # Collect results in original order
        results_dict = {}
        for idx, future in enumerate(as_completed(futures), 1):
            original_idx = futures[future]
            results_dict[original_idx] = future.result()
            print(f"  Reverse [{idx}/{len(tasks)}] Done")
        # Sort by original index to maintain order
        reversed_prompts = [results_dict[i] for i in sorted(results_dict.keys())]

    prompts_dir = Path("app/saved_outputs/prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_file = prompts_dir / f"reversed_prompts_{timestamp}.json"

    with open(prompts_file, 'w', encoding='utf-8') as f:
        json.dump(reversed_prompts, f, ensure_ascii=False, indent=2)

    print(f"  - Saved prompts to {prompts_file}")

    print(f"\nSTEP 2/8: Generating {len(reversed_prompts)} synthetic texts")
    print("-" * 80)

    def generate_task(item):
        original_text = item.get("original_text", "")
        word_count = len(original_text.split()) if original_text else 500
        sentence_count = len([s for s in original_text.split('.') if s.strip()]) if original_text else 25
        prompt = generate_template.format(
            prompt=item["reversed_prompt"],
            word_count=word_count,
            sentence_count=sentence_count
        )
        generated_text = call_gemini(prompt, temperature=0.7)
        return {
            "language": item["language"],
            "subcategory": item["subcategory"],
            "reversed_prompt": item["reversed_prompt"],
            "generated_text": generated_text
        }

    generated_texts = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(generate_task, item): i for i, item in enumerate(reversed_prompts)}
        # Collect results in original order
        results_dict = {}
        for idx, future in enumerate(as_completed(futures), 1):
            original_idx = futures[future]
            results_dict[original_idx] = future.result()
            print(f"  Generate [{idx}/{len(reversed_prompts)}] Done")
        # Sort by original index to maintain order
        generated_texts = [results_dict[i] for i in sorted(results_dict.keys())]

    texts_dir = Path("app/saved_outputs/generated_texts")
    texts_dir.mkdir(parents=True, exist_ok=True)
    texts_file = texts_dir / f"generated_texts_{timestamp}.json"

    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(generated_texts, f, ensure_ascii=False, indent=2)

    print(f"  - Saved texts to {texts_file}")

    print(f"\nSTEP 3/8: Generating UMR graphs for all texts")
    print("-" * 80)

    parser = create_umr_parser()

    # Prepare UMR tasks for human texts
    human_umr_tasks = []
    for item in reversed_prompts:
        human_umr_tasks.append({
            "text": item.get("original_text", ""),
            "language": item["language"],
            "metadata": {
                "source": "human",
                "subcategory": item.get("subcategory", ""),
                "filename": item.get("filename", "")
            }
        })

    # Prepare UMR tasks for generated texts
    llm_umr_tasks = []
    for item in generated_texts:
        llm_umr_tasks.append({
            "text": item.get("generated_text", ""),
            "language": item["language"],
            "metadata": {
                "source": "llm",
                "subcategory": item.get("subcategory", "")
            }
        })

    all_umr_tasks = human_umr_tasks + llm_umr_tasks
    print(f"  Processing {len(human_umr_tasks)} human + {len(llm_umr_tasks)} LLM texts = {len(all_umr_tasks)} total")

    def process_umr_task(task):
        result = parser.parse_text(task["text"], task["language"])
        result.update({
            "original_text": task["text"],
            "language": task["language"],
            "source": task["metadata"]["source"],
            "subcategory": task["metadata"].get("subcategory", ""),
            "filename": task["metadata"].get("filename", "")
        })
        return result

    umr_results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_umr_task, task): i for i, task in enumerate(all_umr_tasks)}
        # Collect results in original order
        results_dict = {}
        for idx, future in enumerate(as_completed(futures), 1):
            original_idx = futures[future]
            result = future.result()
            results_dict[original_idx] = result
            print(f"  UMR [{idx}/{len(all_umr_tasks)}] Done - {result['source']} ({result['language']})")
        # Sort by original index to maintain order
        umr_results = [results_dict[i] for i in sorted(results_dict.keys())]

    # Separate results
    human_umr_results = [r for r in umr_results if r["source"] == "human"]
    llm_umr_results = [r for r in umr_results if r["source"] == "llm"]

    # Save UMR graphs
    umr_dir = Path("app/saved_outputs/umr_graphs")
    umr_dir.mkdir(parents=True, exist_ok=True)
    umr_file = umr_dir / f"umr_graphs_{timestamp}.json"

    umr_data = {
        "human_graphs": human_umr_results,
        "llm_graphs": llm_umr_results,
        "metadata": {
            "total_human": len(human_umr_results),
            "total_llm": len(llm_umr_results),
            "timestamp": timestamp
        }
    }

    with open(umr_file, 'w', encoding='utf-8') as f:
        json.dump(umr_data, f, ensure_ascii=False, indent=2)

    print(f"  - Saved UMR graphs to {umr_file}")

    print(f"\nSTEP 4/8: Visualizing and analyzing UMR graphs")
    print("-" * 80)

    visualizer = UMRVisualizer()

    # Process human graphs
    human_visualizations = []
    for graph_data in human_umr_results:
        viz = visualizer.create_visualization_summary(graph_data)
        viz.update({
            "subcategory": graph_data.get("subcategory", ""),
            "filename": graph_data.get("filename", ""),
            "source": "human"
        })
        human_visualizations.append(viz)

    # Process LLM graphs
    llm_visualizations = []
    for graph_data in llm_umr_results:
        viz = visualizer.create_visualization_summary(graph_data)
        viz.update({
            "subcategory": graph_data.get("subcategory", ""),
            "source": "llm"
        })
        llm_visualizations.append(viz)

    # Compute statistics
    human_stats = {
        "total_graphs": len(human_visualizations),
        "avg_depth": sum(v['structure']['stats']['depth'] for v in human_visualizations if v['success']) / max(len(human_visualizations), 1),
        "avg_concepts": sum(v['structure']['stats']['num_concepts'] for v in human_visualizations if v['success']) / max(len(human_visualizations), 1),
        "avg_roles": sum(v['structure']['stats']['num_roles'] for v in human_visualizations if v['success']) / max(len(human_visualizations), 1),
        "with_aspect": sum(1 for v in human_visualizations if v.get('structure', {}).get('stats', {}).get('has_aspect', False)),
        "with_temporal": sum(1 for v in human_visualizations if v.get('structure', {}).get('stats', {}).get('has_temporal', False))
    }

    llm_stats = {
        "total_graphs": len(llm_visualizations),
        "avg_depth": sum(v['structure']['stats']['depth'] for v in llm_visualizations if v['success']) / max(len(llm_visualizations), 1),
        "avg_concepts": sum(v['structure']['stats']['num_concepts'] for v in llm_visualizations if v['success']) / max(len(llm_visualizations), 1),
        "avg_roles": sum(v['structure']['stats']['num_roles'] for v in llm_visualizations if v['success']) / max(len(llm_visualizations), 1),
        "with_aspect": sum(1 for v in llm_visualizations if v.get('structure', {}).get('stats', {}).get('has_aspect', False)),
        "with_temporal": sum(1 for v in llm_visualizations if v.get('structure', {}).get('stats', {}).get('has_temporal', False))
    }

    # Save visualizations
    viz_dir = Path("app/saved_outputs/umr_visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_file = viz_dir / f"umr_visualizations_{timestamp}.json"

    viz_output = {
        "human_visualizations": human_visualizations,
        "llm_visualizations": llm_visualizations,
        "statistics": {
            "human": human_stats,
            "llm": llm_stats,
            "comparison": {
                "depth_difference": abs(human_stats["avg_depth"] - llm_stats["avg_depth"]),
                "concept_difference": abs(human_stats["avg_concepts"] - llm_stats["avg_concepts"]),
                "role_difference": abs(human_stats["avg_roles"] - llm_stats["avg_roles"])
            }
        }
    }

    with open(viz_file, 'w', encoding='utf-8') as f:
        json.dump(viz_output, f, ensure_ascii=False, indent=2)

    print(f"  - Saved visualizations to {viz_file}")

    print(f"\nSTEP 5/8: Standard UMR analysis (Smatch baseline)")
    print("-" * 80)

    analyzer = create_umr_analyzer()
    comparisons = analyzer.analyze_corpus(human_umr_results, llm_umr_results)
    aggregate_stats = analyzer.calculate_aggregate_stats(comparisons)

    analysis_dir = Path("app/saved_outputs/umr_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = analysis_dir / f"umr_analysis_{timestamp}.json"

    analysis_output = {
        "comparisons": comparisons,
        "aggregate_statistics": aggregate_stats,
        "timestamp": timestamp
    }

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_output, f, ensure_ascii=False, indent=2)

    print(f"  - Saved standard analysis to {analysis_file}")
    print(f"  - Average Smatch F1: {aggregate_stats['avg_smatch_f1']:.4f}")
    print(f"  - Semantic Fidelity: {aggregate_stats['interpretation']['semantic_fidelity']}")
    print(f"  - Complexity Trend: {aggregate_stats['interpretation']['complexity_trend']}")

    print(f"\nSTEP 6/8: Comprehensive multi-metric evaluation")
    print("-" * 80)
    print("  Initializing comprehensive evaluator (loading models)...")

    from app.services.comprehensive_umr_evaluator import create_comprehensive_evaluator

    evaluator = create_comprehensive_evaluator()

    print(f"  Evaluating {len(human_umr_results)} text pairs with:")
    print("    - BERTScore (text-level semantic similarity)")
    print("    - Soft-Smatch (embedding-based concept matching)")
    print("    - Concept Overlap (simple concept set comparison)")
    print("    - Complexity Similarity (structural pattern matching)")

    comprehensive_results = evaluator.evaluate_corpus(human_umr_results, llm_umr_results)
    comprehensive_aggregate = evaluator.compute_aggregate_statistics(comprehensive_results)

    comprehensive_dir = Path("app/saved_outputs/comprehensive_analysis")
    comprehensive_dir.mkdir(parents=True, exist_ok=True)
    comprehensive_file = comprehensive_dir / f"comprehensive_analysis_{timestamp}.json"

    comprehensive_output = {
        "individual_results": comprehensive_results,
        "aggregate_statistics": comprehensive_aggregate,
        "timestamp": timestamp,
        "metadata": {
            "total_pairs": len(comprehensive_results),
            "evaluation_metrics": [
                "BERTScore (text-level semantic similarity)",
                "Standard Smatch (graph isomorphism)",
                "Soft-Smatch (embedding-based concept matching)",
                "Concept Overlap (simple concept set comparison)",
                "Complexity Similarity (structural pattern matching)"
            ]
        }
    }

    with open(comprehensive_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_output, f, ensure_ascii=False, indent=2)

    print(f"  - Saved comprehensive analysis to {comprehensive_file}")
    print(f"\n  COMPREHENSIVE RESULTS:")
    print(f"    - BERTScore F1:        {comprehensive_aggregate['average_bertscore_f1']:.4f} (Target: ≥0.85)")
    print(f"    - Soft-Smatch F1:      {comprehensive_aggregate['average_soft_smatch_f1']:.4f} (Target: ≥0.50)")
    print(f"    - Concept Overlap F1:  {comprehensive_aggregate['average_concept_overlap_f1']:.4f}")
    print(f"    - Standard Smatch F1:  {comprehensive_aggregate['average_standard_smatch_f1']:.4f} (baseline)")
    print(f"    - Complexity Score:    {comprehensive_aggregate['average_complexity_score']:.4f} (Target: ≥0.70)")
    print(f"    - OVERALL SCORE:       {comprehensive_aggregate['average_overall_score']:.4f} (Target: ≥0.70)")
    print(f"    - Interpretation:      {comprehensive_aggregate['interpretation']}")

    print(f"\nSTEP 7/8: Creating comprehensive results file with ALL metrics")
    print("-" * 80)

    all_texts_results = []
    for idx in range(len(human_umr_results)):
        text_result = {
            "text_id": reversed_prompts[idx].get('filename') or reversed_prompts[idx].get('subcategory', f'text_{idx}'),
            "language": reversed_prompts[idx]['language'],
            "subcategory": reversed_prompts[idx]['subcategory'],
            "human_text": reversed_prompts[idx].get('original_text', ''),
            "llm_text": generated_texts[idx].get('generated_text', ''),
            "reversed_prompt": reversed_prompts[idx].get('reversed_prompt', ''),
            "metrics": comprehensive_results[idx],
            "umr_graphs": {
                "human": human_umr_results[idx],
                "llm": llm_umr_results[idx]
            }
        }
        all_texts_results.append(text_result)

    results_dir = Path("app/saved_outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"results_{timestamp}.json"

    complete_results = {
        "timestamp": timestamp,
        "pipeline_summary": {
            "total_texts_processed": len(reversed_prompts),
            "languages": ["romanian", "english"],
            "steps_completed": [
                "1. Reverse Engineering",
                "2. Text Generation",
                "3. UMR Graph Generation",
                "4. Visualization & Analysis",
                "5. Standard UMR Analysis (Smatch)",
                "6. Comprehensive Multi-Metric Evaluation",
                "7. Results Consolidation"
            ]
        },
        "all_texts": all_texts_results,
        "aggregate_statistics": {
            "comprehensive_metrics": comprehensive_aggregate,
            "standard_analysis": aggregate_stats,
            "human_umr_stats": human_stats,
            "llm_umr_stats": llm_stats,
            "visualization_statistics": viz_output["statistics"]
        },
        "files_generated": {
            "prompts": str(prompts_file),
            "texts": str(texts_file),
            "umr_graphs": str(umr_file),
            "visualizations": str(viz_file),
            "analysis": str(analysis_file),
            "comprehensive": str(comprehensive_file),
            "results": str(results_file)
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, ensure_ascii=False, indent=2)

    print(f"  - Saved complete results to {results_file}")
    print(f"  - Total metrics per text: BERTScore + Smatch + Soft-Smatch + Concept Overlap + Complexity")
    print(f"  - File contains ALL data for {len(all_texts_results)} texts")

    print(f"\nSTEP 8/8: Pipeline summary")
    print("-" * 80)
    print(f"  Human texts processed: {len(reversed_prompts)}")
    print(f"  LLM texts generated: {len(generated_texts)}")
    print(f"  UMR graphs created: {len(umr_results)}")
    print(f"  \nHuman UMR Statistics:")
    print(f"    - Avg depth: {human_stats['avg_depth']:.2f}")
    print(f"    - Avg concepts: {human_stats['avg_concepts']:.2f}")
    print(f"    - Avg roles: {human_stats['avg_roles']:.2f}")
    print(f"    - With aspect: {human_stats['with_aspect']}/{human_stats['total_graphs']}")
    print(f"    - With temporal: {human_stats['with_temporal']}/{human_stats['total_graphs']}")
    print(f"  \nLLM UMR Statistics:")
    print(f"    - Avg depth: {llm_stats['avg_depth']:.2f}")
    print(f"    - Avg concepts: {llm_stats['avg_concepts']:.2f}")
    print(f"    - Avg roles: {llm_stats['avg_roles']:.2f}")
    print(f"    - With aspect: {llm_stats['with_aspect']}/{llm_stats['total_graphs']}")
    print(f"    - With temporal: {llm_stats['with_temporal']}/{llm_stats['total_graphs']}")

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETE - ALL RESULTS SAVED")
    print(f"Master results file: {results_file}")
    print("=" * 80)

    return {
        "reversed_prompts": reversed_prompts,
        "generated_texts": generated_texts,
        "umr_graphs": {
            "human": human_umr_results,
            "llm": llm_umr_results
        },
        "statistics": viz_output["statistics"],
        "semantic_analysis": aggregate_stats,
        "comprehensive_analysis": comprehensive_aggregate,
        "files": {
            "prompts": str(prompts_file),
            "texts": str(texts_file),
            "umr_graphs": str(umr_file),
            "visualizations": str(viz_file),
            "analysis": str(analysis_file),
            "comprehensive": str(comprehensive_file),
            "results": str(results_file)
        },
        "counts": {
            "human_texts": len(reversed_prompts),
            "llm_texts": len(generated_texts),
            "total_umr_graphs": len(umr_results)
        }
    }


@app.post("/generate-umr-graphs/{dataset_name}")
def generate_umr_graphs(dataset_name, input_file: str = None):
    """
    Generate UMR graphs for both human-written and LLM-generated texts.
    """
    print("Starting UMR graph generation")
    
    if dataset_name == "multipleye":

        # Load the reversed prompts file (contains original texts)
        if not input_file:
            prompts_dir = Path("app/saved_outputs/prompts")
            prompt_files = sorted(prompts_dir.glob("reversed_prompts_*.json"))
            if not prompt_files:
                return {"error": "No reversed prompts files found"}
            input_file = str(prompt_files[-1])

        print(f"Loading data from: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

        # Also load generated texts if available
        texts_dir = Path("app/saved_outputs/generated_texts")
        text_files = sorted(texts_dir.glob("generated_texts_*.json"))
        generated_data = []

        if text_files:
            print(f"Loading generated texts from: {text_files[-1]}")
            with open(text_files[-1], 'r', encoding='utf-8') as f:
                generated_data = json.load(f)

        # Create UMR parser instance
        parser = create_umr_parser()

        # Prepare tasks for human texts
        human_tasks = []
        for item in prompts_data:
            human_tasks.append({
                "text": item.get("original_text", ""),
                "language": item["language"],
                "metadata": {
                    "source": "human",
                    "subcategory": item.get("subcategory", ""),
                    "filename": item.get("filename", "")
                }
            })

        # Prepare tasks for generated texts
        generated_tasks = []
        for item in generated_data:
            generated_tasks.append({
                "text": item.get("generated_text", ""),
                "language": item["language"],
                "metadata": {
                    "source": "llm",
                    "subcategory": item.get("subcategory", ""),
                    "reversed_prompt": item.get("reversed_prompt", "")
                }
            })

        all_tasks = human_tasks + generated_tasks
        total = len(all_tasks)

        print(f"Processing {len(human_tasks)} human texts and {len(generated_tasks)} generated texts")
        print(f"Total: {total} UMR graphs to generate")

        def process_umr_task_multipleye(task):
            result = parser.parse_text(task["text"], task["language"])
            result.update({
                "original_text": task["text"],
                "language": task["language"],
                "source": task["metadata"]["source"],
                "subcategory": task["metadata"].get("subcategory", ""),
                "filename": task["metadata"].get("filename", "")
            })
            return result

        results = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_umr_task_multipleye, task): i for i, task in enumerate(all_tasks)}
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{idx}/{total}] Done - {result['source']} ({result['language']})")

        # Separate human and LLM results
        human_results = [r for r in results if r["source"] == "human"]
        llm_results = [r for r in results if r["source"] == "llm"]

        # Save results
        save_dir = Path("app/saved_outputs/umr_graphs")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = save_dir / f"umr_graphs_{timestamp}.json"

        output_data = {
            "human_graphs": human_results,
            "llm_graphs": llm_results,
            "metadata": {
                "total_human": len(human_results),
                "total_llm": len(llm_results),
                "timestamp": timestamp,
                "source_file": input_file
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_file}")
        print("UMR graph generation complete")

        return {
            "human_graphs": human_results,
            "llm_graphs": llm_results,
            "count": {
                "human": len(human_results),
                "llm": len(llm_results),
                "total": total
            },
            "saved_to": str(output_file)
        }
        
    elif dataset_name == "zuco2":
        df_zuco = load_zuco2_data()
        # Generate UMR graphs for Zuco2 dataset
        df_zuco = load_zuco2_data()

        # Create UMR parser instance
        parser = create_umr_parser()

        # Prepare tasks for Zuco2 sentences
        zuco_tasks = []
        for _, row in df_zuco.iterrows():
            zuco_tasks.append({
                "text": row['sentence'],
                "language": "english",
                "metadata": {
                    "source": "zuco2",
                    "sentence_id": row['sentence_id']
                }
            })

        total = len(zuco_tasks)
        print(f"Processing {total} Zuco2 sentences")

        def process_umr_task_zuco2(task):
            result = parser.parse_text(task["text"], task["language"])
            result.update({
                "text": task["text"],
                "language": task["language"],
                "source": task["metadata"]["source"],
                "sentence_id": task["metadata"]["sentence_id"]
            })
            return result

        results = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_umr_task_zuco2, task): i for i, task in enumerate(zuco_tasks)}
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
            print(f"[{idx}/{total}] Done - Sentence ID: {result['sentence_id']}")

        # Save results
        save_dir = Path("app/saved_outputs/umr_graphs")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = save_dir / f"umr_graphs_zuco2_{timestamp}.json"

        output_data = {
            "zuco2_graphs": results,
            "metadata": {
                "total_sentences": len(results),
                "timestamp": timestamp
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_file}")
        print("UMR graph generation for Zuco2 complete")

        return {
            "zuco2_graphs": results,
            "count": len(results),
            "saved_to": str(output_file)
        }


@app.post("/visualize-umr-graphs")
def visualize_umr_graphs(dataset_name, input_file: str = None):
    """
    Visualize and analyze UMR graphs from a saved file.
    """
    print("Starting UMR graph visualization")

    if not input_file:
        umr_dir = Path("app/saved_outputs/umr_graphs")
        if not umr_dir.exists():
            return {"error": "No UMR graphs directory found. Generate graphs first."}

        umr_files = sorted(umr_dir.glob("umr_graphs_*.json"))
        if not umr_files:
            return {"error": "No UMR graph files found. Generate graphs first."}
        input_file = str(umr_files[-1])

    print(f"Loading UMR graphs from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if dataset_name == "multipleye":

        human_graphs = data.get("human_graphs", [])
        llm_graphs = data.get("llm_graphs", [])

        # Create visualizer
        visualizer = UMRVisualizer()

        # Process human graphs
        human_visualizations = []
        for graph_data in human_graphs:
            viz = visualizer.create_visualization_summary(graph_data)
            viz.update({
                "subcategory": graph_data.get("subcategory", ""),
                "filename": graph_data.get("filename", ""),
                "source": "human"
            })
            human_visualizations.append(viz)

        # Process LLM graphs
        llm_visualizations = []
        for graph_data in llm_graphs:
            viz = visualizer.create_visualization_summary(graph_data)
            viz.update({
                "subcategory": graph_data.get("subcategory", ""),
                "source": "llm"
            })
            llm_visualizations.append(viz)

        # Compute aggregate statistics
        human_stats = {
            "total_graphs": len(human_visualizations),
            "avg_depth": sum(v['structure']['stats']['depth'] for v in human_visualizations if v['success']) / max(len(human_visualizations), 1),
            "avg_concepts": sum(v['structure']['stats']['num_concepts'] for v in human_visualizations if v['success']) / max(len(human_visualizations), 1),
            "avg_roles": sum(v['structure']['stats']['num_roles'] for v in human_visualizations if v['success']) / max(len(human_visualizations), 1),
            "with_aspect": sum(1 for v in human_visualizations if v.get('structure', {}).get('stats', {}).get('has_aspect', False)),
            "with_temporal": sum(1 for v in human_visualizations if v.get('structure', {}).get('stats', {}).get('has_temporal', False))
        }

        llm_stats = {
            "total_graphs": len(llm_visualizations),
            "avg_depth": sum(v['structure']['stats']['depth'] for v in llm_visualizations if v['success']) / max(len(llm_visualizations), 1),
            "avg_concepts": sum(v['structure']['stats']['num_concepts'] for v in llm_visualizations if v['success']) / max(len(llm_visualizations), 1),
            "avg_roles": sum(v['structure']['stats']['num_roles'] for v in llm_visualizations if v['success']) / max(len(llm_visualizations), 1),
            "with_aspect": sum(1 for v in llm_visualizations if v.get('structure', {}).get('stats', {}).get('has_aspect', False)),
            "with_temporal": sum(1 for v in llm_visualizations if v.get('structure', {}).get('stats', {}).get('has_temporal', False))
        }

        # Save visualization results
        save_dir = Path("app/saved_outputs/umr_visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = save_dir / f"umr_visualizations_{timestamp}.json"

        output_data = {
            "human_visualizations": human_visualizations,
            "llm_visualizations": llm_visualizations,
            "statistics": {
                "human": human_stats,
                "llm": llm_stats,
                "comparison": {
                    "depth_difference": abs(human_stats["avg_depth"] - llm_stats["avg_depth"]),
                    "concept_difference": abs(human_stats["avg_concepts"] - llm_stats["avg_concepts"]),
                    "role_difference": abs(human_stats["avg_roles"] - llm_stats["avg_roles"])
                }
            },
            "source_file": input_file
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_file}")
        print("Visualization complete")

        return {
            "statistics": output_data["statistics"],
            "sample_human": human_visualizations[:3] if human_visualizations else [],
            "sample_llm": llm_visualizations[:3] if llm_visualizations else [],
            "saved_to": str(output_file)
        }
        
    elif dataset_name == "zuco2":

        zuco_graphs = data.get("zuco2_graphs", [])

        # Create visualizer
        visualizer = UMRVisualizer()

        # Process Zuco2 graphs
        zuco_visualizations = []
        for graph_data in zuco_graphs:
            viz = visualizer.create_visualization_summary(graph_data)
            viz.update({
                "sentence_id": graph_data.get("sentence_id", ""),
                "source": "zuco2"
            })
            zuco_visualizations.append(viz)

        # Save visualization results
        save_dir = Path("app/saved_outputs/umr_visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = save_dir / f"umr_visualizations_zuco2_{timestamp}.json"

        output_data = {
            "zuco2_visualizations": zuco_visualizations,
            "source_file": input_file
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_file}")
        print("Visualization complete")

        return {
            "sample_zuco2": zuco_visualizations[:3] if zuco_visualizations else [],
            "saved_to": str(output_file)
        }


@app.post("/analyze-umr-semantics")
def analyze_umr_semantics(input_file: str = None):
    print("Starting UMR semantic analysis")

    if not input_file:
        umr_dir = Path("app/saved_outputs/umr_graphs")
        if not umr_dir.exists():
            return {"error": "No UMR graphs directory found. Generate graphs first."}

        umr_files = sorted(umr_dir.glob("umr_graphs_*.json"))
        if not umr_files:
            return {"error": "No UMR graph files found. Generate graphs first."}
        input_file = str(umr_files[-1])

    print(f"Loading UMR graphs from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    human_graphs = data.get("human_graphs", [])
    llm_graphs = data.get("llm_graphs", [])

    if len(human_graphs) != len(llm_graphs):
        return {"error": f"Mismatch: {len(human_graphs)} human graphs vs {len(llm_graphs)} LLM graphs"}

    print(f"Analyzing {len(human_graphs)} graph pairs")

    analyzer = create_umr_analyzer()
    comparisons = analyzer.analyze_corpus(human_graphs, llm_graphs)

    aggregate_stats = analyzer.calculate_aggregate_stats(comparisons)

    save_dir = Path("app/saved_outputs/umr_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = save_dir / f"umr_analysis_{timestamp}.json"

    output_data = {
        "comparisons": comparisons,
        "aggregate_statistics": aggregate_stats,
        "source_file": input_file,
        "timestamp": timestamp
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_file}")
    print("Semantic analysis complete")

    print("\n=== AGGREGATE RESULTS ===")
    print(f"  Average Smatch F1: {aggregate_stats['avg_smatch_f1']}")
    print(f"  High Fidelity (>0.80): {aggregate_stats['high_fidelity_count']}/{aggregate_stats['total_pairs']}")
    print(f"  Low Fidelity (<0.60): {aggregate_stats['low_fidelity_count']}/{aggregate_stats['total_pairs']}")
    print(f"  Semantic Fidelity: {aggregate_stats['interpretation']['semantic_fidelity']}")
    print(f"  Complexity Trend: {aggregate_stats['interpretation']['complexity_trend']}")

    return {
        "aggregate_statistics": aggregate_stats,
        "sample_comparisons": comparisons[:5],
        "total_analyzed": len(comparisons),
        "saved_to": str(output_file)
    }
    
    
@app.post("/compare-umr-with-eye-tracking")
def compare_umr_with_eye_tracking(umr_input_file: str = None, eye_tracking_avg_sentence_file: str = None, eye_tracking_part_sentence_file: str = None):
    print("Starting comparison of UMR graphs with eye-tracking data")

    # Load UMR graphs
    if not umr_input_file:
        umr_dir = Path("app/saved_outputs/umr_graphs")
        if not umr_dir.exists():
            return {"error": "No UMR graphs directory found. Generate graphs first."}

        umr_files = sorted(umr_dir.glob("umr_graphs_*.json"))
        if not umr_files:
            return {"error": "No UMR graph files found. Generate graphs first."}
        umr_input_file = str(umr_files[-1])

    print(f"Loading UMR graphs from: {umr_input_file}")
    with open(umr_input_file, 'r', encoding='utf-8') as f:
        umr_data = json.load(f)
        
    umr_graphs = umr_data.get("zuco2_graphs", [])
        
    # Get UMR sentence-level stats
    umr_analyzer = create_umr_analyzer()
    sentence_umr_stats = umr_analyzer.analyze_sentences(umr_graphs)
    print(f"Analyzed UMR statistics for {len(sentence_umr_stats)} sentences")
    
    # Convert UMR stats to DataFrame
    df_umr = pd.DataFrame.from_dict(sentence_umr_stats, orient="index")
    df_umr.index.name = "sentence_id"
    df_umr = df_umr.reset_index()
    
    # Load sentence-level average eye-tracking data
    if not eye_tracking_avg_sentence_file:
        eye_tracking_avg_sentence_file = "app/data/zuco2_average_sentence_level.csv"

    if not Path(eye_tracking_avg_sentence_file).exists():
        return {"error": "Average sentence-level eye-tracking file not found."}
    
    df_eye_avg = pd.read_csv(eye_tracking_avg_sentence_file)
    
    # Load sentence-level participant eye-tracking data
    if not eye_tracking_part_sentence_file:
        eye_tracking_part_sentence_file = "app/data/zuco2_participants_sentence_level.csv"
        
    if not Path(eye_tracking_part_sentence_file).exists():
        return {"error": "Participant sentence-level eye-tracking file not found."}
    
    df_eye_part = pd.read_csv(eye_tracking_part_sentence_file)
    
    comparator = UMREyeTrackingComparator(df_umr, df_eye_avg, df_eye_part)

    # Average sentence-level correlations
    result_avg = comparator.compute_correlations(return_heatmap=True, participant_level=False)

    # Participant sentence-level correlations
    if df_eye_part is not None:
        result_participant = comparator.compute_correlations(return_heatmap=False, participant_level=True)
        participant_summary = result_participant["participant_level_summary"]
        participant_plot_b64 = comparator.plot_participant_correlations(participant_summary)
    else:
        participant_summary = None
        
    # Mixed-effects modelling
    result = comparator.mixed_effects_model(["num_nodes", "num_edges"], "FFD_avg")
    print(result["model_summary"])

    return {
        "top_sentence_level_correlations": result_avg["correlations_sorted"][:10],
        "sentence_level_corr_matrix": result_avg["corr_matrix"],
        "participant_level_summary": participant_summary
    }


@app.post("/analyze-umr-semantics-comprehensive")
def analyze_umr_semantics_comprehensive(input_file: str = None):
    print("=" * 80)
    print("STARTING COMPREHENSIVE UMR SEMANTIC EVALUATION")
    print("Multi-Metric Evaluation Framework (BERTScore + Soft-Smatch + Complexity)")
    print("=" * 80)

    if not input_file:
        umr_dir = Path("app/saved_outputs/umr_graphs")
        if not umr_dir.exists():
            return {"error": "No UMR graphs directory found. Generate graphs first."}

        umr_files = sorted(umr_dir.glob("umr_graphs_*.json"))
        if not umr_files:
            return {"error": "No UMR graph files found. Generate graphs first."}
        input_file = str(umr_files[-1])

    print(f"Loading UMR graphs from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    human_graphs = data.get("human_graphs", [])
    llm_graphs = data.get("llm_graphs", [])

    if len(human_graphs) != len(llm_graphs):
        return {"error": f"Mismatch: {len(human_graphs)} human graphs vs {len(llm_graphs)} LLM graphs"}

    print(f"Loaded {len(human_graphs)} human graphs and {len(llm_graphs)} LLM graphs")

    from app.services.comprehensive_umr_evaluator import create_comprehensive_evaluator

    print("Initializing comprehensive evaluator (loading models)...")
    evaluator = create_comprehensive_evaluator()

    print(f"\nEvaluating {len(human_graphs)} text pairs with multi-metric analysis...")
    print("-" * 80)

    comprehensive_results = evaluator.evaluate_corpus(human_graphs, llm_graphs)

    print("\nComputing aggregate statistics...")
    aggregate = evaluator.compute_aggregate_statistics(comprehensive_results)

    save_dir = Path("app/saved_outputs/comprehensive_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = save_dir / f"comprehensive_analysis_{timestamp}.json"

    output_data = {
        "individual_results": comprehensive_results,
        "aggregate_statistics": aggregate,
        "timestamp": timestamp,
        "source_file": input_file,
        "metadata": {
            "total_pairs": len(comprehensive_results),
            "evaluation_metrics": [
                "BERTScore (text-level semantic similarity)",
                "Standard Smatch (graph isomorphism)",
                "Soft-Smatch (embedding-based concept matching)",
                "Concept Overlap (simple concept set comparison)",
                "Complexity Similarity (structural pattern matching)"
            ]
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved comprehensive analysis to: {output_file}")
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Average BERTScore F1:        {aggregate['average_bertscore_f1']:.4f} (Target: ≥0.85)")
    print(f"Average Soft-Smatch F1:      {aggregate['average_soft_smatch_f1']:.4f} (Target: ≥0.50)")
    print(f"Average Concept Overlap F1:  {aggregate['average_concept_overlap_f1']:.4f}")
    print(f"Average Standard Smatch F1:  {aggregate['average_standard_smatch_f1']:.4f} (baseline)")
    print(f"Average Complexity Score:    {aggregate['average_complexity_score']:.4f} (Target: ≥0.70)")
    print(f"\nOVERALL SCORE:               {aggregate['average_overall_score']:.4f} (Target: ≥0.70)")
    print(f"INTERPRETATION:              {aggregate['interpretation']}")
    print("\nDistribution:")
    print(f"  Excellent (>0.80): {aggregate['distribution']['excellent']} texts")
    print(f"  Good (0.70-0.80):  {aggregate['distribution']['good']} texts")
    print(f"  Acceptable (0.60-0.70): {aggregate['distribution']['acceptable']} texts")
    print(f"  Poor (<0.60):      {aggregate['distribution']['poor']} texts")
    print("\nSuccess Criteria:")
    for criterion, target in aggregate['success_criteria'].items():
        if 'achieved' in criterion:
            status = "✓" if target else "✗"
            print(f"  {status} {criterion.replace('_', ' ').title()}: {target}")
    print("=" * 80)

    return {
        "aggregate_statistics": aggregate,
        "sample_results": comprehensive_results[:3],
        "total_analyzed": len(comprehensive_results),
        "saved_to": str(output_file)
    }