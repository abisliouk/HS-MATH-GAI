import openai
import json
import time
from pathlib import Path
from uuid import uuid4
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.keys import PREMIUM_API_KEY
from scripts.const import OUTPUT_DIR_COT, INPUT_PATH_ORIGINAL, MODEL_LOCAL
from scripts.cot.utils import evaluate_confidence_accuracy_cot, get_prompt_cot, call_api_cot, safe_parse_cot_json


# Configuration
NUM_SAMPLES = None  # Set to an integer for testing on fewer samples. None for all.
OUTPUT_FILENAME = "prediction_with_uncertainties_cot.json"
OUTPUT_PATH = f"{OUTPUT_DIR_COT}/{OUTPUT_FILENAME}"

client = openai.OpenAI(api_key=PREMIUM_API_KEY, base_url="http://10.227.119.44:8000/v1")

# Load dataset
with open(INPUT_PATH_ORIGINAL, "r") as f:
    all_data = json.load(f)
    dataset = all_data if NUM_SAMPLES is None else all_data[:NUM_SAMPLES]

results = []

for idx, item in enumerate(dataset):
    prompt = get_prompt_cot(item['question_en'])
    raw_response = call_api_cot(client, prompt, model=MODEL_LOCAL, idx=idx)

    if not raw_response:
        continue

    parsed = safe_parse_cot_json(raw_response, idx=idx)
    if parsed is None or not isinstance(parsed, dict):
        print(f"[Warning @ idx={idx}]: Failed to parse JSON.")
        continue

    steps = parsed.get("steps", [])
    final_conf = parsed.get("final_confidence", {})
    pred = parsed.get("predicted_answer", "")

    results.append({
        "id": item.get("id", str(uuid4())),
        "question": item["question_en"],
        "expected_answer": item["answer"],
        "model_response": {
            "predicted_answer": pred,
            "intermediate_confidences": [
                {
                    "step_number": step.get("step_number"),
                    "self_eval_confidence": step.get("self_confidence", 0.0),
                    "logit_based_confidence": step.get("confidence_distribution", {}).get(pred, 0.0),
                    "internal_based_confidence": step.get("internal_confidence", 0.0)
                } for step in steps
            ],
            "final_confidence": {
                "self_eval_confidence": final_conf.get("self_confidence", 0.0),
                "logit_based_confidence": final_conf.get("confidence_distribution", {}).get(pred, 0.0),
                "internal_based_confidence": final_conf.get("internal_confidence", 0.0)
            }
        },
        "raw_text": raw_response,
        "timestamp": time.time()
    })

    Path(OUTPUT_DIR_COT).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

# Final save
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

# Confidence-Accuracy Evaluation
confidence_keys = [
    ("final_confidence.self_confidence", "confidence_accuracy_self_eval_cot.json"),
    ("final_confidence.logit_based_confidence", "confidence_accuracy_logit_cot.json"),
    ("final_confidence.internal_confidence", "confidence_accuracy_internal_cot.json"),
]

evaluate_confidence_accuracy_cot(results, confidence_keys, output_dir=OUTPUT_DIR_COT)
