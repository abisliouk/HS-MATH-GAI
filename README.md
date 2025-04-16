# HS-MATH-GAI: Evaluating and Enhancing High School-Level Mathematical Reasoning in LLMs

This project evaluates AI-generated answers to high school math problems using **GPT models** and computes **uncertainty quantification (UQ)** metrics. It supports **both original and augmented datasets**, and computes confidence-accuracy reliability tables.

---

## 📁 Project Structure

```
.
├── data/                          # Input JSON datasets (original and augmented)
│   ├── reformatted_augmented_data.json
│   └── math_translated_scored.json
├── outputs/                       # Stores raw predictions + confidence-accuracy tables
│   ├── prediction_with_uncertainties.json
│   ├── confidence_accuracy_self_eval.json
│   ├── confidence_accuracy_logit.json
│   └── confidence_accuracy_internal.json
├── scripts/
│   ├── evaluate_original.py       # Inference on original math problems
│   ├── evaluate_augmented.py      # Inference on augmented math problems
│   ├── utils.py                   # Prompting, parsing, API and UQ evaluation functions
│   ├── const.py                   # Constants and config
│   └── keys.py                    # Contains your `PREMIUM_API_KEY`
```

---

## ✅ Setup

1. Install dependencies:

```bash
pip install openai numpy
```

2. Create `scripts/keys.py`:

```python
PREMIUM_API_KEY = "sk-..."  # Your actual key here
```

3. Configure `const.py` for:
   - API model: `MODEL_LOCAL` (e.g., `Qwen-7B-Chat`)
   - Input/output paths:
     - `INPUT_PATH_ORIGINAL = "data/math_translated_scored.json"`
     - `INPUT_PATH_AUGMENTED = "data/reformatted_augmented_data.json"`
     - `OUTPUT_DIR = "outputs"`

---

## 🚀 Usage

### 1. Evaluate Original Math Problems

Run the following:

```bash
python scripts/evaluate_original.py
```

This will:

- Use questions from `data/math_translated_scored.json`
- Query the model and store structured responses (answer + 3 confidences)
- Save results to `outputs/prediction_with_uncertainties.json`
- Generate confidence-accuracy reliability tables:
  - `confidence_accuracy_self_eval.json`
  - `confidence_accuracy_logit.json`
  - `confidence_accuracy_internal.json`

---

### 2. Evaluate Augmented Math Problems

Run:

```bash
python scripts/evaluate_augmented.py
```

This will:

- Process the three augmented questions from each sample in `reformatted_augmented_data.json`
- Store each result with fields: `id`, `question_version`, and confidence metrics
- Save outputs and reliability plots to `outputs/`

---

## 📊 Confidence Types

Each prediction includes:

| Confidence Field            | Description                                                     |
|----------------------------|-----------------------------------------------------------------|
| `self_eval_confidence`     | Model's subjective estimate of correctness                      |
| `logit_based_confidence`   | Probability assigned to the predicted answer                    |
| `internal_based_confidence`| Confidence based on internal consistency (self-declared)        |

These are used to generate **confidence-accuracy** reliability tables.

---

## 📈 Outputs

Each evaluated entry (original or augmented) looks like:

```json
{
  "id": "abc-123",
  "question": "...",
  "expected_answer": ["D"],
  "model_response": {
    "predicted_answer": "D",
    "confidence": {
      "self_eval_confidence": 0.92,
      "logit_based_confidence": 0.87,
      "internal_based_confidence": 0.88
    }
  },
  ...
}
```

Reliability tables (binned by confidence):

```json
[
  {"confidence_bin": "0.8-0.9", "num_samples": 45, "accuracy": 0.844},
  ...
]
```

---

## 📌 Notes

- All prompts are strict JSON-only with clear format expectations
- Invalid responses are logged and skipped
- Model's confidence distribution must sum to 1
- Use `NUM_SAMPLES` in `evaluate_*.py` to limit batch size during testing

---

## 🧠 Authors

Created as part of **EEL6935: Safe Autonomous Systems** coursework.

---

## 🔐 Disclaimer

You must have a valid OpenAI API key and/or access to the local server endpoint to run the scripts.