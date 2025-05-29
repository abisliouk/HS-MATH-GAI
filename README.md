
# HS-MATH-LLM: Evaluating and Enhancing High School-Level Mathematical Reasoning in LLMs

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project evaluates the ability of large language models (LLMs) to solve high school-level math problems. It focuses on reasoning accuracy and confidence calibration using both single-step (baseline) and step-by-step (Chain-of-Thought, CoT) prompting.

For each model response, the system extracts structured outputs including predicted answers and multiple confidence estimates. These values quantify how certain the model is in its answers and are binned into reliability tables to evaluate calibration — i.e., whether higher confidence corresponds to higher accuracy.

---

## 📁 Project Structure

```
.
├── data/                          # Input datasets (original and reformatted)
│   ├── original_dataset.json
│   └── reformatted_augmented_dataset.json
│
├── outputs_baseline/             # Results from baseline evaluation
│   ├── prediction_with_uncertainties.json
│   ├── confidence_accuracy_self_eval.json
│   ├── confidence_accuracy_logit.json
│   └── confidence_accuracy_internal.json
│
├── outputs_cot/                  # Results from Chain-of-Thought evaluation
│   ├── prediction_with_uncertainties_cot.json
│   ├── confidence_accuracy_self_eval_cot.json
│   ├── confidence_accuracy_logit_cot.json
│   └── confidence_accuracy_internal_cot.json
│
├── scripts/
│   ├── baseline/
│   │   ├── evaluate_original.py       # Baseline evaluation script
│   │   └── utils.py                   # Baseline utils (prompting, parsing, evaluation)
│   │
│   ├── cot/
│   │   ├── evaluate_original_cot.py   # CoT evaluation script
│   │   └── utils.py                   # CoT-specific prompt and evaluation utilities
│   │
│   ├── const.py                       # Shared constants and configuration
│   ├── keys.py                        # Contains your PREMIUM_API_KEY
```

---

## ✅ Setup

To get started, follow these steps:

1. **Install dependencies**

Make sure you have Python 3.8+ and run:

```bash
pip install openai numpy
```

2. **Add your API key**

Create a file at `scripts/keys.py` containing:

```python
PREMIUM_API_KEY = "sk-..."  # Replace with your actual API key
```

3. **Check configuration**

Open `scripts/const.py` and verify or update the following paths and model name:

```python
MODEL_GPT_3_5 = "gpt-3.5-turbo"
INPUT_PATH_ORIGINAL = "data/original_dataset.json"
INPUT_PATH_AUGMENTED = "data/reformatted_augmented_dataset.json"
OUTPUT_DIR_BASELINE = "outputs_baseline"
OUTPUT_DIR_COT = "outputs_cot"
```

4. **(Optional) Limit number of samples**

To test on a subset of data, set `NUM_SAMPLES = <integer>` at the top of:

- `scripts/baseline/evaluate_original.py`
- `scripts/cot/evaluate_original_cot.py`

---

## 🚀 Usage

### 1. Evaluate Original Math Problems (Baseline)

To evaluate the original dataset using a single-step baseline approach:

```bash
python scripts/baseline/evaluate_original.py
```

This script will:

- Load questions from `data/original_dataset.json`
- Generate standard prompts using `scripts/utils.py`
- Query the GPT model defined in `scripts/const.py` (default: `gpt-3.5-turbo`)
- Parse structured outputs including:
  - Predicted final answer
  - Final confidence metrics:
    - `self_eval_confidence`
    - `logit_based_confidence`
    - `internal_based_confidence`
- Save results to `outputs_baseline/prediction_with_uncertainties.json`
- Create confidence-accuracy reliability tables:
  - `confidence_accuracy_self_eval.json`
  - `confidence_accuracy_logit.json`
  - `confidence_accuracy_internal.json`

To test with fewer examples, set `NUM_SAMPLES` to an integer at the top of `evaluate_original.py`.

#### 🧾 Example Output Snippet for Baseline

```json
{
  "id": "xyz-789",
  "question": "If x + 2 = 5, what is x?",
  "expected_answer": ["C"],
  "model_response": {
    "predicted_answer": "C",
    "confidence": {
      "self_eval_confidence": 0.93,
      "logit_based_confidence": 0.91,
      "internal_based_confidence": 0.89
    }
  },
  "raw_text": "...",
  "timestamp": 1717019999.0
}
```

### 2. Evaluate Math Problems with Chain of Thought (CoT) Approach

To evaluate mathematical reasoning using a step-by-step Chain of Thought strategy:

```bash
python scripts/cot/evaluate_original_cot.py
```

This script will:

- Load questions from `data/original_dataset.json`
- Generate CoT-formatted prompts using `scripts/cot/utils.py`
- Query the GPT model defined in `scripts/const.py` (default: `gpt-3.5-turbo`)
- Parse structured outputs including:
  - Predicted final answer
  - Intermediate reasoning steps
  - Confidence metrics at each step:
    - `self_eval_confidence`
    - `logit_based_confidence`
    - `internal_based_confidence`
- Save results to `outputs_cot/prediction_with_uncertainties_cot.json`
- Create confidence-accuracy reliability tables:
  - `confidence_accuracy_self_eval_cot.json`
  - `confidence_accuracy_logit_cot.json`
  - `confidence_accuracy_internal_cot.json`

To test with fewer examples, set `NUM_SAMPLES` to an integer at the top of `evaluate_original_cot.py`.

#### 🧾 Example Output Snippet for CoT

```json
{
  "id": "abc-123",
  "question": "What is the value of 2x when x = 3?",
  "expected_answer": ["B"],
  "model_response": {
    "predicted_answer": "B",
    "intermediate_confidences": [
      {
        "step_number": 1,
        "self_eval_confidence": 0.85,
        "logit_based_confidence": 0.80,
        "internal_based_confidence": 0.75
      },
      {
        "step_number": 2,
        "self_eval_confidence": 0.90,
        "logit_based_confidence": 0.85,
        "internal_based_confidence": 0.80
      }
    ],
    "final_confidence": {
      "self_eval_confidence": 0.95,
      "logit_based_confidence": 0.90,
      "internal_based_confidence": 0.92
    }
  },
  "raw_text": "...",
  "timestamp": 1717012345.0
}
```


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

### 📊 Reliability Tables (Binned by Confidence)

After each evaluation, the script generates reliability tables that show the accuracy of model predictions within different confidence intervals. These tables are saved as JSON files in the corresponding output directory (e.g., `outputs_baseline/` or `outputs_cot/`).

Each entry in the table represents:

- A confidence bin (e.g., 0.6–0.7)
- The number of predictions that fell within that bin
- The proportion of those predictions that were correct (accuracy)

#### 🔢 Example

```json
[
  {
    "confidence_bin": "0.0-0.1",
    "num_samples": 3,
    "accuracy": 0.0
  },
  {
    "confidence_bin": "0.8-0.9",
    "num_samples": 45,
    "accuracy": 0.844
  },
  {
    "confidence_bin": "0.9-1.0",
    "num_samples": 82,
    "accuracy": 0.902
  }
]
```

Use these tables to analyze how well model confidence correlates with actual correctness (calibration).

---

---

## 📌 Additional Notes

- Prompts strictly enforce valid JSON-only output.
- Invalid or unparsable responses are logged and skipped during evaluation.
- The `"confidence_distribution"` field must sum to 1.
- Intermediate results are saved incrementally to prevent data loss on interruption.
- Set `NUM_SAMPLES` in each evaluation script to quickly test on smaller subsets.

---

## 🧠 Authors

Created as part of **EEL6935: Safe Autonomous Systems** coursework at University of Florida.

---

## 🔐 Disclaimer

You must have a valid OpenAI API key and/or access to the local server endpoint to run the scripts.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
