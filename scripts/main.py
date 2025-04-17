import json
import time
import re
import openai

# ─── CONFIG ────────────────────────────────────────────────────
openai.api_key  = "EMPTY"    # or set your real key here
openai.api_base = "http://10.227.119.44:8000/v1"
MODEL           = "gpt-4o-mini"

INPUT_FILE      = "data.json"
OUTPUT_FILE     = "augmented_qa.json"
FAILURES_FILE   = "failures.json"

VARIANTS_PER    = 3
MAX_RETRIES     = 5
RATE_LIMIT_SEC  = 1.0

SYSTEM_PROMPT = f"""
You are an expert math tutor.  Reply _only_ with a JSON array of exactly {VARIANTS_PER} objects.
Each object must have exactly these keys:
  - "question": string
  - "options": object with keys "A","B","C","D" mapping to strings
  - "answer": one of "A","B","C","D"

No markdown, no extra keys, no narration—just the JSON array.
""".strip()

# ─── UTILITIES ──────────────────────────────────────────────────
def extract_first_json_array(text: str):
    """
    Finds the first balanced `[...]` block in `text`, escapes stray backslashes,
    and returns the parsed JSON array.
    """
    start = text.find('[')
    if start < 0:
        raise ValueError("No '[' found in LLM output")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                blob = text[start:i+1]
                # Escape any backslash not part of a valid JSON escape
                blob = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', blob)
                return json.loads(blob)
    raise ValueError("Unbalanced brackets in LLM output")

def generate_variants(question: str, answer: str):
    """
    Calls the LLM up to MAX_RETRIES times, extracting a clean JSON array each time.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": f"Original question:\n{question}\n\nOriginal answer: {answer}\nGenerate {VARIANTS_PER} variants."}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content
        try:
            return extract_first_json_array(raw)
        except Exception as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] parse error: {e}")
            time.sleep(RATE_LIMIT_SEC)
    raise RuntimeError("All attempts to parse JSON failed")

# ─── MAIN ───────────────────────────────────────────────────────
def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        originals = json.load(f)

    augmented = []
    failures  = []

    print(f"Loaded {len(originals)} questions. Beginning augmentation…")
    for idx, entry in enumerate(originals, start=1):
        q = entry.get("question", "").strip()
        a = entry.get("answer",   "").strip()
        print(f"\n▶ #{idx}/{len(originals)}: {q[:60]}… (Ans: {a})")

        try:
            variants = generate_variants(q, a)
            augmented.append({
                "question": q,
                "answer":   a,
                "variants": variants
            })
            print("   ✓ success")
        except Exception as err:
            print(f"   ✗ error: {err}")
            failures.append({
                "index":    idx,
                "question": q,
                "answer":   a,
                "error":    str(err)
            })

        time.sleep(RATE_LIMIT_SEC)

    # write successes
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(augmented, out, indent=2, ensure_ascii=False)
    print(f"\n✅ Wrote {len(augmented)} augmented entries to {OUTPUT_FILE}")

    # write failures
    with open(FAILURES_FILE, "w", encoding="utf-8") as out:
        json.dump(failures, out, indent=2, ensure_ascii=False)
    print(f"ℹ️  Logged {len(failures)} failures to {FAILURES_FILE}")

if __name__ == "__main__":
    main()




# import json
# import time
# import openai
# import re

# # ─── CONFIG ────────────────────────────────────────────────────
# openai.api_key  = "EMPTY"
# openai.api_base = "http://10.227.119.44:8000/v1"
# MODEL          = "gpt-4o-mini"

# INPUT_FILE     = "data.json"
# OUTPUT_FILE    = "augmented_debug.json"
# FAILURES_FILE  = "failures_debug.json"

# VARIANTS_PER   = 3
# MAX_RETRIES    = 3
# RATE_LIMIT_SEC = 1.0

# SYSTEM_PROMPT = f"""
# You are an expert math tutor. Reply _only_ with a JSON array of exactly {VARIANTS_PER} objects.
# Each object must have exactly these keys—and nothing else:
#   • "question": string
#   • "options": {{ "A": string, "B": string, "C": string, "D": string }}
#   • "answer": one of "A","B","C","D"
# No markdown, no commentary, no extra keys—just the raw array.
# """.strip()

# def extract_first_json_array(text: str):
#     # find the first '[' then scan forward matching braces until the array closes
#     start = text.find('[')
#     if start < 0:
#         raise ValueError("no '[' found")
#     depth = 0
#     for i in range(start, len(text)):
#         if text[i] == '[':
#             depth += 1
#         elif text[i] == ']':
#             depth -= 1
#             if depth == 0:
#                 snippet = text[start:i+1]
#                 return json.loads(snippet)
#     raise ValueError("unbalanced brackets")

# def generate_variants(question: str, answer: str):
#     for attempt in range(1, MAX_RETRIES+1):
#         resp = openai.ChatCompletion.create(
#             model=MODEL,
#             messages=[
#                 {"role": "system",  "content": SYSTEM_PROMPT},
#                 {"role": "user",    "content": f"Q: {question}\nA: {answer}\nGenerate {VARIANTS_PER} variants."}
#             ],
#             temperature=0.3,
#             max_tokens=600
#         )
#         raw = resp.choices[0].message.content
#         try:
#             return extract_first_json_array(raw)
#         except Exception as e:
#             print(f"  [Attempt {attempt}] parse error: {e}")
#             print("  Raw response was:\n", raw, "\n" + "-"*60)
#             time.sleep(RATE_LIMIT_SEC)
#     raise RuntimeError("All retries failed")

# def main():
#     data = json.load(open(INPUT_FILE, encoding="utf-8"))
#     augmented, failures = [], []

#     # for debugging, only do the first 10. Change to `for entry in data:` to run all.
#     for idx, entry in enumerate(data[:10], start=1):
#         q = entry["question"].strip()
#         a = entry["answer"].strip()
#         print(f"\n▶ #{idx}: {q[:80]}… (Ans: {a})")
#         try:
#             variants = generate_variants(q, a)
#             augmented.append({
#                 "question": q,
#                 "answer": a,
#                 "variants": variants
#             })
#             print("   ✓ OK")
#         except Exception as err:
#             print("   ✗ Failed:", err)
#             failures.append({
#                 "index": idx,
#                 "question": q,
#                 "answer": a,
#                 "error": str(err)
#             })
#         time.sleep(RATE_LIMIT_SEC)

#     json.dump(augmented, open(OUTPUT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
#     json.dump(failures,  open(FAILURES_FILE,"w", encoding="utf-8"), indent=2, ensure_ascii=False)
#     print(f"\n✅ Wrote {len(augmented)} augmented entries to {OUTPUT_FILE}")
#     print(f"ℹ️   Logged {len(failures)} failures to {FAILURES_FILE}")

# if __name__ == "__main__":
#     main()