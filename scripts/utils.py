import json
import numpy as np
from collections import defaultdict
import json
from pathlib import Path


def safe_parse_json(raw_response, idx=None):
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        if idx is not None:
            print(f"[Warning @ idx={idx}]: Failed to parse JSON. Raw output:\n{raw_response}")
        return None


def get_prompt(question):
    return f"""
You are a careful math tutor. Here is the high school math problem: {question}.

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


def call_api(client, prompt, model, idx=None):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a PhD-level mathematician. Return only valid JSON without any explanations, markdown or extra text."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Error @ idx={idx}]: {e}")
        return None


def evaluate_confidence_accuracy(results, confidence_keys, output_dir="outputs"):
    """
    Calculates and saves accuracy for each confidence bin for the specified confidence types.

    Args:
        results: List of prediction result dicts.
        confidence_keys: List of (key, output_filename) pairs.
        output_dir: Directory where result JSONs will be saved.
    """

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

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / output_file
        with open(path, "w") as f:
            json.dump(table, f, indent=2)

        print(f"✅ Saved {conf_key} confidence-accuracy table to: {path}")

