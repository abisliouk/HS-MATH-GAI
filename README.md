# HS-MATH-GAI: Evaluating and Enhancing High School-Level Mathematical Reasoning in LLMs

## Project Summary

This project introduces **HS-MATH-GAI**, a framework to systematically evaluate and improve the **symbolic mathematical reasoning** abilities of Large Language Models (LLMs) on **high school-level math problems**.

The project focuses on three main components:
1. **Benchmark Construction** – Developing a symbolic math benchmark aligned with high school curricula.
2. **Uncertainty Quantification** – Measuring confidence and variability in LLM-generated reasoning chains.
3. **Reasoning Workflow Optimization** – Enhancing reasoning robustness via purification and step-wise refinement techniques.

## 🔍 Math Reasoning Evaluation with Uncertainty Quantification (UQ)

This project implements an automated pipeline for evaluating high school math reasoning performance using OpenAI's GPT models. It also incorporates **four Uncertainty Quantification (UQ)** methods to better estimate the model’s confidence and reliability on each problem.

---
## 📂 Project Structure

- The `data/` directory contains the input dataset file `math_translated_scored.json`, which holds English-translated math problems along with their ground truth answers.

- The `scripts/` directory includes the main evaluation script `evaluate.py`. This script queries the OpenAI API to solve problems and compute multiple uncertainty quantification (UQ) scores.

- The `outputs/` directory is used to store the evaluation results. Specifically, the file `eval_structured_with_uq_optimized.json` contains the model responses, predicted answers, and four types of UQ confidence scores for each evaluated math problem.

- The `README.md` file you're reading now provides setup instructions, usage, and project documentation.

## 🚀 Features

- Automatically solves math problems with GPT-3.5 Turbo.
- Returns:
  - Step-by-step reasoning
  - Final answer (A/B/C/D)
  - Confidence estimates from four UQ methods:
    - ✅ Self-Evaluation (verbal score from model)
    - 🔢 Logit-Based (via softmax-like distribution over answers)
    - 🧠 Internal-Based (reasoning clarity, approximated)
- Saves structured results to a JSON file.
- Designed for reproducible experimentation.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:abisliouk/EEL6935-HS-MATH-GAI.git
cd EEL6935-HS-MATH-GAI
```

### 2. Create Python Environment

Make sure you have Python 3.8+ installed.

```bash
pip install openai
```

### 3. Set OpenAI API Key

To use the OpenAI API, you need an API key from your account.

1. Go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).
2. Click **Create new secret key**.
3. Copy the generated key.

Then, set the key in your terminal session using:

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

This will:
- Load problems from `data/math_translated_scored.json`.
- Send each question to GPT-3.5 Turbo using structured prompts.
- Collect predictions, reasoning, and four types of uncertainty:
  - **Self-evaluation confidence**: Model's own declared certainty.
  - **Logit-based confidence**: Max softmax probability over A–D.
  - **Internal-based confidence**: Self-estimated based on reasoning coherence.
- Save structured results to:
  - `outputs/eval_structured_with_uq_optimized.json`

> ⏳ Evaluation may take several minutes depending on your OpenAI quota and the number of samples configured in the script.


### 📊 Output Format
```bash
{
  "id": "...",
  "question": "...",
  "expected_answer": ["D"],
  "model_response": {
    "reasoning": "...",
    "predicted_answer": "D",
    "confidence": {
      "self_eval_confidence": 0.87,
      "logit_based_confidence": 0.9,
      "internal_based_confidence": 0.85
    }
  },
  "raw_text": "...",
  "timestamp": 1744131300.123456
}
```