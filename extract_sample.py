import json
import sys

with open('app/saved_outputs/umr_graphs/umr_graphs_20260110_213621.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

human_sample = data['human_graphs'][5]
llm_sample = data['llm_graphs'][5]

sample_data = {
    "human": {
        "filename": human_sample.get('filename', 'N/A'),
        "language": human_sample.get('language'),
        "text": human_sample.get('original_text', ''),
        "umr_graph": human_sample.get('umr_graph', '')
    },
    "llm": {
        "language": llm_sample.get('language'),
        "text": llm_sample.get('original_text', ''),
        "umr_graph": llm_sample.get('umr_graph', '')
    }
}

with open('sample_pair.json', 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

print("Sample extracted successfully!")
print(f"Human file: {sample_data['human']['filename']}")
print(f"Language: {sample_data['human']['language']}")
print(f"Human text length: {len(sample_data['human']['text'])} chars")
print(f"LLM text length: {len(sample_data['llm']['text'])} chars")
print(f"Human UMR length: {len(sample_data['human']['umr_graph'])} chars")
print(f"LLM UMR length: {len(sample_data['llm']['umr_graph'])} chars")
