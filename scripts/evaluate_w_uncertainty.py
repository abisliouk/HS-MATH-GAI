import openai
import json
import time
import os
from pathlib import Path
from uuid import uuid4
from statistics import mode

# Configuration
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.5
NUM_SAMPLES = 5
CONSISTENCY_SAMPLES = 5
INPUT_PATH = "data/math_translated_scored.json"
OUTPUT_PATH = "outputs/eval_structured_with_uq_optimized.json"

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dataset
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)[:NUM_SAMPLES]

results = []

for idx, item in enumerate(dataset):
    prompt = f"""
You are a careful math tutor. Solve the following high school math problem step-by-step.
Then:
1. Choose your final answer (A, B, C, or D).
2. Estimate your **self-confidence** (how likely you think your answer is correct) ∈ [0.0, 1.0].
3. Estimate your **internal confidence** (Use internal states (e.g., hidden activations, attention heads) to estimate confidence or uncertainty.) ∈ [0.0, 1.0].
4. Provide your **confidence distribution** over choices A–D as JSON (e.g. {{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}}).

Problem:
{item['question_en']}

Return only a JSON with the following fields:
{{
  "reasoning": "...",
  "predicted_answer": "A",
  "self_confidence": 0.93,
  "internal_confidence": 0.91,
  "confidence_distribution": {{
    "A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4
  }}
}}
"""

    def call_api(p):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": p}
                ],
                temperature=TEMPERATURE
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

    # Consistency-based UQ
    answers = []
    for _ in range(CONSISTENCY_SAMPLES):
        retry_raw = call_api(prompt)
        try:
            retry_parsed = json.loads(retry_raw)
            ans = retry_parsed.get("predicted_answer")
            if ans:
                answers.append(ans)
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

    # Final output
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
