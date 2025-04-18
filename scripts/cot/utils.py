import re
import numpy as np
import json
from pathlib import Path
from collections import defaultdict


def call_api_cot(client, prompt, model, idx=None):
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
- Do not copy the example JSON — fill the values based on your actual solution process.
- Do not include anything outside [JSON_START] and [JSON_END].
"""

def evaluate_confidence_accuracy_cot(results, confidence_keys, output_dir="outputs"):
    """
    Evaluates and saves accuracy bins for final_confidence fields in CoT-style outputs.

    Args:
        results: List of CoT-format result dictionaries.
        confidence_keys: List of tuples (nested confidence key path, output filename).
        output_dir: Directory to save output JSONs.
    """
    bins = np.arange(0.0, 1.1, 0.1)

    def get_nested_value(d, key_path):
        """Access nested keys like 'final_confidence.self_eval_confidence' safely."""
        keys = key_path.split(".")
        for key in keys:
            d = d.get(key, {})
        return d if isinstance(d, (int, float)) else 0.0

    for conf_key, output_file in confidence_keys:
        bin_counts = defaultdict(int)
        bin_correct = defaultdict(int)

        for r in results:
            conf = get_nested_value(r["model_response"], conf_key)
            correct = r["model_response"]["predicted_answer"] in r["expected_answer"]

            for i in range(len(bins) - 1):
                if bins[i] <= conf < bins[i + 1] or (conf == 1.0 and bins[i + 1] == 1.0):
                    bin_key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                    bin_counts[bin_key] += 1
                    bin_correct[bin_key] += int(correct)
                    break

        # Format results
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

        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / output_file
        with open(path, "w") as f:
            json.dump(table, f, indent=2)

        print(f"✅ Saved {conf_key} confidence-accuracy table to: {path}")
