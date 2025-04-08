import openai
import json
import time
import os
from pathlib import Path
from uuid import uuid4

# Configuration
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0
INPUT_PATH = "data/math_translated_scored.json"
OUTPUT_PATH = "outputs/eval_structured_responses.json"
NUM_SAMPLES = 10
SAVE_EVERY = 1

# Create OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dataset
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)[:NUM_SAMPLES]

results = []

for idx, item in enumerate(dataset):
    prompt = f"""
Solve the following high school math problem. Provide your step-by-step reasoning, final answer (one of the options: A, B, C or D), and a confidence estimate between 0 and 1.

Problem:
{item['question_en']}

Respond in JSON format like:
{{
  "reasoning": "...",
  "predicted_answer": "A",
  "confidence": 0.92
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a careful math tutor. Answer in structured JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )

        content = response.choices[0].message.content.strip()
        try:
            structured = json.loads(content)
        except Exception as parse_err:
            structured = {
                "reasoning": content,
                "predicted_answer": None,
                "confidence": None,
                "parse_error": str(parse_err)
            }

        results.append({
            "id": item.get("id", str(uuid4())),
            "question": item["question"],
            "expected_answer": item["answer"],
            "model_response": structured,
            "raw_text": content,
            "timestamp": time.time()
        })

        if idx % SAVE_EVERY == 0 or idx == NUM_SAMPLES - 1:
            Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

        time.sleep(1.0)

    except Exception as e:
        print(f"[Error @ idx={idx}]: {e}")

# Final save
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
