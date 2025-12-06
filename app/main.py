from fastapi import FastAPI
from app.loaders import load_multipleye_data
from app.services import call_gemini
from app.utils.load_yaml_prompts import load_prompt
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()


@app.get("/load-data")
def load_data():
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


@app.post("/reverse-engineer")
def reverse_engineer():
    print("Starting reverse engineering")
    df_ro, df_en = load_multipleye_data()

    prompt_template = load_prompt("app/prompts/reverse_engineer.yaml")

    tasks = []
    for _, row in df_ro.iterrows():
        tasks.append({
            "language": "romanian",
            "subcategory": row['subcategory'],
            "filename": row['filename'],
            "original_text": row['text'],
            "prompt": prompt_template.format(text=row['text'])
        })

    for _, row in df_en.iterrows():
        tasks.append({
            "language": "english",
            "subcategory": row['subcategory'],
            "original_text": row.get('text', ''),
            "prompt": prompt_template.format(text=row['text'])
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
    with ThreadPoolExecutor(max_workers=50) as executor:
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
def generate_texts(prompts_data: dict):
    print("Starting text generation")
    prompt_template = load_prompt("app/prompts/generate_text.yaml")

    items = prompts_data.get("prompts", [])
    total = len(items)

    print(f"Generating {total} texts concurrently")

    def process_item(item):
        prompt = prompt_template.format(prompt=item["reversed_prompt"])
        generated_text = call_gemini(prompt)
        return {
            "language": item["language"],
            "subcategory": item["subcategory"],
            "reversed_prompt": item["reversed_prompt"],
            "generated_text": generated_text
        }

    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
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
def full_pipeline():
    print("Starting full pipeline")
    df_ro, df_en = load_multipleye_data()

    reverse_template = load_prompt("app/prompts/reverse_engineer.yaml")
    generate_template = load_prompt("app/prompts/generate_text.yaml")

    tasks = []
    for _, row in df_ro.iterrows():
        tasks.append({
            "language": "romanian",
            "subcategory": row['subcategory'],
            "filename": row['filename'],
            "original_text": row['text'],
            "prompt": reverse_template.format(text=row['text'])
        })

    for _, row in df_en.iterrows():
        tasks.append({
            "language": "english",
            "subcategory": row['subcategory'],
            "original_text": row.get('text', ''),
            "prompt": reverse_template.format(text=row['text'])
        })

    print(f"Step 1: Reverse engineering {len(tasks)} texts")

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
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(reverse_task, task): i for i, task in enumerate(tasks)}
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            reversed_prompts.append(result)
            print(f"Reverse [{idx}/{len(tasks)}] Done")

    prompts_dir = Path("app/saved_outputs/prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_file = prompts_dir / f"reversed_prompts_{timestamp}.json"

    with open(prompts_file, 'w', encoding='utf-8') as f:
        json.dump(reversed_prompts, f, ensure_ascii=False, indent=2)

    print(f"Saved prompts to {prompts_file}")

    print(f"Step 2: Generating {len(reversed_prompts)} texts")

    def generate_task(item):
        prompt = generate_template.format(prompt=item["reversed_prompt"])
        generated_text = call_gemini(prompt)
        return {
            "language": item["language"],
            "subcategory": item["subcategory"],
            "reversed_prompt": item["reversed_prompt"],
            "generated_text": generated_text
        }

    generated_texts = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(generate_task, item): i for i, item in enumerate(reversed_prompts)}
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            generated_texts.append(result)
            print(f"Generate [{idx}/{len(reversed_prompts)}] Done")

    texts_dir = Path("app/saved_outputs/generated_texts")
    texts_dir.mkdir(parents=True, exist_ok=True)
    texts_file = texts_dir / f"generated_texts_{timestamp}.json"

    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(generated_texts, f, ensure_ascii=False, indent=2)

    print(f"Saved texts to {texts_file}")
    print("Full pipeline complete")

    return {
        "reversed_prompts": reversed_prompts,
        "generated_texts": generated_texts,
        "count": len(generated_texts),
        "prompts_file": str(prompts_file),
        "texts_file": str(texts_file)
    }
