import openai
import json
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
from collections import defaultdict
import re

from const import PREMIUM_API_KEY

# Constants
SELF_EVAL_CONFIDENCE = "self_eval_confidence"
LOGIT_BASED_CONFIDENCE = "logit_based_confidence"
INTERNAL_BASED_CONFIDENCE = "internal_based_confidence"

# Configuration
MODEL_FREE = "gpt-3.5-turbo"
MODEL_LOCAL = "Qwen-7B-Chat"
NUM_SAMPLES = None # For processing only a few samples instead of the entire dataset
INPUT_PATH = "data/reformatted_augmented_data.json"
OUTPUT_PATH = "outputs/prediction_with_uncertainties.json"

client = openai.OpenAI(api_key=PREMIUM_API_KEY, base_url="http://10.227.119.44:8000/v1")

# Load dataset
with open(INPUT_PATH, "r") as f:
    all_data = json.load(f)
    dataset = all_data if NUM_SAMPLES is None else all_data[:NUM_SAMPLES]

results = []

def safe_parse_json(raw_response, idx=None):
    try:
        # Remove code fencing
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Fix smart quotes or special characters
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # Try parsing normally
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed

    except Exception as e:
        if idx is not None:
            print(f"[Warning @ idx={idx}]: Failed to parse JSON. Raw output:\n{raw_response}")
        return None



for idx, item in enumerate(dataset):
    prompt = f"""
You are a careful math tutor. Here is the high school math problem: {item['question_en']}.

Respond with **only valid JSON** using the following format, do not include explanations, markdown or extra text:

{{
  "predicted_answer": "Your answer here (A, B, C, or D)",
  "self_confidence": float ∈ [0.0, 1.0],
  "internal_confidence": float ∈ [0.0, 1.0],
  "confidence_distribution": {{
    "A": float ∈ [0.0, 1.0],
    "B": float ∈ [0.0, 1.0],
    "C": float ∈ [0.0, 1.0],
    "D": float ∈ [0.0, 1.0]
  }}
}}

Where:
1. "predicted_answer" - your final answer (A, B, C, or D).
2. "self_confidence" - Estimation of your self-confidence (how likely you think your answer is correct).
3. "internal_confidence" - Estimation of your internal confidence (based on reasoning clarity and certainty).
4. "confidence_distribution" - your confidence distribution over choices A–D.

Requirements:
- Your answer must be valid JSON.
- Make sure the confidence values sum to 1 in "confidence_distribution".
- Do not copy the example — fill the values based on your actual solution process.
- Only include the JSON object, with no additional comments or markdown.
"""

    def call_api(p):
        try:
            response = client.chat.completions.create(
                model=MODEL_LOCAL,
                messages=[
                    {"role": "system", "content": "You are a PhD-level mathematician. Return only valid JSON without any explanations, markdown or extra text."},
                    {"role": "user", "content": p}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Error @ idx={idx}]: {e}")
            return None

    # Primary response
    raw_response = call_api(prompt)
    if not raw_response:
        continue

    parsed = safe_parse_json(raw_response, idx=idx)
    if parsed is None:
        print(f"[Warning @ idx={idx}]: Failed to parse JSON.")
        continue

    predicted = parsed.get("predicted_answer")
    self_conf = float(parsed.get("self_confidence", 0.0))
    internal_conf = float(parsed.get("internal_confidence", 0.0))
    dist = parsed.get("confidence_distribution", {})
    logit_based = dist.get(predicted, 0.0)

    # Final output
    results.append({
        "id": item.get("id", str(uuid4())),
        "question": item["question"],
        "expected_answer": item["answer"],
        "model_response": {
            "predicted_answer": predicted,
            "confidence": {
                SELF_EVAL_CONFIDENCE: round(self_conf, 3),
                LOGIT_BASED_CONFIDENCE: round(logit_based, 3),
                INTERNAL_BASED_CONFIDENCE: round(internal_conf, 3)
            }
        },
        "raw_text": raw_response,
        "timestamp": time.time()
    })

    # Save after each step
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

# Final save
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

    

# Confidence-accuracy evaluation

# List of confidence fields to evaluate
confidence_keys = [
    (SELF_EVAL_CONFIDENCE, "confidence_accuracy_self_eval.json"),
    (LOGIT_BASED_CONFIDENCE, "confidence_accuracy_logit.json"),
    (INTERNAL_BASED_CONFIDENCE, "confidence_accuracy_internal.json")
]

bins = np.arange(0.0, 1.1, 0.1)

for conf_key, output_file in confidence_keys:
    bin_counts = defaultdict(int)
    bin_correct = defaultdict(int)

    for r in results:
        conf = r["model_response"]["confidence"].get(conf_key, 0.0)
        correct = r["model_response"]["predicted_answer"] in r["expected_answer"]

        for i in range(len(bins) - 1):
            if bins[i] <= conf < bins[i + 1] or (conf == 1.0 and bins[i + 1] == 1.0):
                bin_key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                bin_counts[bin_key] += 1
                bin_correct[bin_key] += int(correct)
                break

    # Build table
    table = []
    for bin_key in sorted(bin_counts.keys()):
        total = bin_counts[bin_key]
        correct = bin_correct[bin_key]
        acc = correct / total if total > 0 else 0.0
        table.append({
            "confidence_bin": bin_key,
            "num_samples": total,
            "accuracy": round(acc, 3)
        })

    # Save table
    output_path = f"outputs/{output_file}"
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"✅ Saved {conf_key} confidence-accuracy table to: {output_path}")

