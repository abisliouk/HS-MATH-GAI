import openai
import json
from pathlib import Path

# Config
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0
INPUT_PATH = "data/math_translated_scored.json"

# Use environment variable (make sure OPENAI_API_KEY is set)
client = openai.OpenAI()

# Load first sample
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

sample = dataset[0]
prompt = f"Solve the following high school math problem:\n\n{sample['question_en']}\n\nPlease provide the final answer at the end."

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a math tutor. Solve the problem step-by-step."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    output_text = response.choices[0].message.content
    print("\n✅ Prompt:\n", prompt)
    print("\n🧠 GPT-3.5 Response:\n", output_text)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
