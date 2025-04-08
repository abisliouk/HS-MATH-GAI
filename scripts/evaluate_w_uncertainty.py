import openai
import json
import time
import os
from pathlib import Path
from uuid import uuid4
from statistics import mode

# Configuration
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
NUM_SAMPLES = 5
CONSISTENCY_SAMPLES = 5
INPUT_PATH = "data/math_translated_scored.json"
OUTPUT_PATH = "outputs/eval_structured_with_uq.json"

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)[:NUM_SAMPLES]

results = []

for idx, item in enumerate(dataset):
    base_prompt = f"""
Solve the following high school math problem. Provide step-by-step reasoning and final answer (only A, B, C, or D).

Problem:
{item['question_en']}

Respond ONLY in JSON format like:
{{
  "reasoning": "...",
  "predicted_answer": "A"
}}
"""

    def get_response(prompt):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a careful math tutor. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Error @ idx={idx}]: {e}")
            return None

    # Main response
    content = get_response(base_prompt)
    if not content:
        continue

    try:
        parsed = json.loads(content)
    except Exception as parse_err:
        parsed = {
            "reasoning": content,
            "predicted_answer": None,
            "parse_error": str(parse_err)
        }

    predicted = parsed.get("predicted_answer")

    # Self-confidence prompt
    self_eval_prompt = f"""
Given your previous answer to this problem:

{item['question_en']}

How confident are you in your final answer on a scale from 0.0 to 1.0? Only return a float.
"""
    try:
        self_conf_raw = get_response(self_eval_prompt)
        self_conf = float(self_conf_raw.strip())
    except:
        self_conf = 0.0

    # Logit-based confidence prompt (Softmax entropy estimate)
    logit_prompt = f"""
Estimate your confidence distribution (Softmax probability) over the answer options A, B, C, and D for the following question:

{item['question_en']}

Return JSON like:
{{"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}}
"""
    try:
        logit_dist_text = get_response(logit_prompt)
        logit_dist = json.loads(logit_dist_text)
        logit_based = max(logit_dist.values())
    except:
        logit_based = 0.0

    # Internal-based confidence prompt
    internal_prompt = f"""
For the math problem below, estimate your internal certainty (based on your reasoning clarity and internal consistency) as a float from 0.0 to 1.0:

{item['question_en']}

Return only a float.
"""
    try:
        internal_raw = get_response(internal_prompt)
        internal_conf = float(internal_raw.strip())
    except:
        internal_conf = 0.0

    # Consistency-based confidence
    answers = []
    for _ in range(CONSISTENCY_SAMPLES):
        sample_text = get_response(base_prompt)
        try:
            parsed_sample = json.loads(sample_text)
            answer = parsed_sample.get("predicted_answer")
            if answer:
                answers.append(answer)
        except:
            continue

    if answers:
        try:
            consensus = mode(answers)
            agreement_ratio = answers.count(consensus) / len(answers)
        except:
            agreement_ratio = 0.0
    else:
        agreement_ratio = 0.0

    results.append({
        "id": item.get("id", str(uuid4())),
        "question": item["question"],
        "expected_answer": item["answer"],
        "model_response": {
            "reasoning": parsed.get("reasoning", ""),
            "predicted_answer": predicted,
            "confidence": {
                "self_eval_confidence": round(self_conf, 3),
                "consistency_based_confidence": round(agreement_ratio, 3),
                "logit_based_confidence": round(logit_based, 3),
                "internal_based_confidence": round(internal_conf, 3)
            }
        },
        "raw_text": content,
        "timestamp": time.time()
    })

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    time.sleep(1.2)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)