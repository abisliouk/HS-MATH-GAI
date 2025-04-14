import openai
import json
import time
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
from collections import defaultdict

from const import PREMIUM_API_KEY

# Constants
SELF_EVAL_CONFIDENCE = "self_eval_confidence"
LOGIT_BASED_CONFIDENCE = "logit_based_confidence"
INTERNAL_BASED_CONFIDENCE = "internal_based_confidence"

# Configuration
MODEL_FREE = "gpt-3.5-turbo"
MODEL_LOCAL = "Qwen-7B-Chat"
NUM_SAMPLES = 3 # For processing only a few samples instead of the entire dataset
INPUT_PATH = "data/math_translated_scored.json"
OUTPUT_PATH = "outputs/prediction_with_uncertainties.json"

client = openai.OpenAI(api_key=PREMIUM_API_KEY, base_url="http://10.227.119.44:8000/v1")

# Load dataset
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)[:NUM_SAMPLES]

results = []

for idx, item in enumerate(dataset):
    prompt = f"""
You are a careful math tutor. Solve the following high school math problem step-by-step providing the following:
1. "reasoning" - Explain your solution step by step.
2. "predicted_answer" - Provide your final answer (A, B, C, or D) based on your "reasoning".
3. "self_confidence" - Estimate your self-confidence (how likely you think your answer is correct) ∈ [0.0, 1.0].
4. "internal_confidence" - Estimate your internal confidence (based on reasoning clarity and certainty) ∈ [0.0, 1.0].
5. "confidence_distribution" - Provide your confidence distribution over choices A–D as JSON (e.g. {{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}}).

Problem:
{item['question_en']}

Return only a JSON with the following fields and corresponding data types:
{{
  "reasoning": string,
  "predicted_answer": enum["A", "B", "C", "D"],
  "self_confidence": float ∈ [0.0, 1.0],
  "internal_confidence": float ∈ [0.0, 1.0],
  "confidence_distribution": {{
        "A": float ∈ [0.0, 1.0],
        "B": float ∈ [0.0, 1.0],
        "C": float ∈ [0.0, 1.0],
        "D": float ∈ [0.0, 1.0]
    }}
}}
"""

    def call_api(p):
        try:
            response = client.chat.completions.create(
                model=MODEL_LOCAL,
                messages=[
                    {"role": "system", "content": "You are a PhD-level mathematician."},
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

    try:
        parsed = json.loads(raw_response)
    except Exception as parse_err:
        parsed = {
            "reasoning": raw_response,
            "predicted_answer": None,
            "self_confidence": 0.0,
            "internal_confidence": 0.0,
            "confidence_distribution": {},
            "parse_error": str(parse_err)
        }

    predicted = parsed.get("predicted_answer")
    self_conf = float(parsed.get("self_confidence", 0.0))
    internal_conf = float(parsed.get("internal_confidence", 0.0))
    dist = parsed.get("confidence_distribution", {})
    logit_based = max(dist.values()) if dist else 0.0

    # Final output
    results.append({
        "id": item.get("id", str(uuid4())),
        "question": item["question"],
        "expected_answer": item["answer"],
        "model_response": {
            "reasoning": parsed.get("reasoning", ""),
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

    time.sleep(1.0)

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

