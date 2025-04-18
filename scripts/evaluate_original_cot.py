import openai
import json
import time
from pathlib import Path
from uuid import uuid4

from keys import PREMIUM_API_KEY
from const import *
from utils import evaluate_confidence_accuracy
import re

# Configuration
NUM_SAMPLES = 3  # Set to an integer for testing on fewer samples. None for all.
OUTPUT_FILENAME = "prediction_with_uncertainties_cot.json"
OUTPUT_PATH = f"{OUTPUT_DIR}/{OUTPUT_FILENAME}"

client = openai.OpenAI(api_key=PREMIUM_API_KEY, base_url="http://10.227.119.44:8000/v1")

# Load dataset
with open(INPUT_PATH_ORIGINAL, "r") as f:
    all_data = json.load(f)
    dataset = all_data if NUM_SAMPLES is None else all_data[:NUM_SAMPLES]

results = []


def call_api(client, prompt, model, idx=None):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a PhD-level mathematician. Answer must be ONLY a valid JSON, with no explanations, comments or markdown."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Error @ idx={idx}]: {e}")
        return None

def safe_parse_cot_json(raw_response, idx=None):
    try:
        # Extract between [JSON_START] and [JSON_END]
        match = re.search(r"\[JSON_START\](.*?)\[JSON_END\]", raw_response, re.DOTALL)
        if not match:
            raise ValueError("JSON markers not found.")

        json_text = match.group(1).strip()

        # Normalize smart quotes and remove potential trailing junk
        json_text = json_text.replace("\u201c", '"').replace("\u201d", '"') \
                             .replace("\u2018", "'").replace("\u2019", "'")

        # Parse and validate structure
        parsed = json.loads(json_text)

        if not isinstance(parsed, dict):
            raise ValueError("Top-level response is not a dict.")
        for field in ["steps", "final_confidence", "predicted_answer"]:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        return parsed

    except Exception as e:
        if idx is not None:
            print(f"[Warning @ idx={idx}]: Failed to parse CoT JSON. Raw output:\n{raw_response}")
        return None




def get_prompt_cot(question):
    return f"""
You are a careful math tutor solving the following high school problem using a step-by-step approach.

Problem:
{question}

Your task is:
1. Break the solution into 3 clear, numbered steps.
2. For each step, estimate:
   - self_confidence (how likely you think this step is correct) ∈ [0.0, 1.0]
   - internal_confidence (how clear and certain your internal reasoning is) ∈ [0.0, 1.0]
   - confidence_distribution over A–D options at this step ∈ [0.0, 1.0] summing to 1.0

3. After the final step, provide your final answer and repeat confidence estimates.

Output the result as ONLY a JSON list using this structure exactly (no explanations, no markdown, no commentary) wrapped between `[JSON_START]` and `[JSON_END]`:

[JSON_START]
{{
  "steps": [
    {{
      "step_number": 1,
      "self_confidence": 0.7,
      "internal_confidence": 0.8,
      "confidence_distribution": {{
        "A": 0.2,
        "B": 0.2,
        "C": 0.3,
        "D": 0.3
      }}
    }},
    ...
  ],
  "predicted_answer": "C",
  "final_confidence": {{
    "self_confidence": 0.95,
    "internal_confidence": 0.95,
    "confidence_distribution": {{
      "A": 0.05,
      "B": 0.0,
      "C": 0.1,
      "D": 0.85
    }}
  }}
}}
[JSON_END]

Requirements for the output:
- Your answer must be ONLY a valid JSON, with no additional explanations, comments or markdown.
- Do not copy the example — fill the values based on your actual solution process.
- Do not include anything outside [JSON_START] and [JSON_END].
"""

for idx, item in enumerate(dataset):
    prompt = get_prompt_cot(item['question_en'])
    raw_response = call_api(client, prompt, model=MODEL_LOCAL, idx=idx)

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

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

# Final save
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

# Confidence-Accuracy Evaluation
confidence_keys = [
    ("final_confidence.self_eval_confidence", "confidence_accuracy_self_eval_cot.json"),
    ("final_confidence.logit_based_confidence", "confidence_accuracy_logit_cot.json"),
    ("final_confidence.internal_based_confidence", "confidence_accuracy_internal_cot.json")
]

evaluate_confidence_accuracy(results, confidence_keys, output_dir=OUTPUT_DIR, confidence_key_path=True)
