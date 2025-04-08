# Member 3 – Implementation Plan

This README documents the complete implementation workflow for **Member 3** in the Safe Autonomous Systems project. It follows the project timeline and tasks described in the preliminary proposal.

---

## ✅ Task Overview

### 🔹 April 6 – Automation & Baseline Uncertainty Quantification

#### 1. Set up an Automation Testing Workflow
- **Goal**: Automatically evaluate AI-generated math solutions on a test set.
- **Steps**:
  1. **Prepare input/output format**:
     - Inputs: Math problems (text), expected outputs (answers).
     - Outputs: Predicted answer, reasoning (if CoT), confidence (if applicable).
  2. **Write `evaluate.py`**:
     - Loads the test dataset.
     - Sends each problem to the model API.
     - Saves raw responses (JSON/text) locally.
  3. **Use `openai.ChatCompletion.create()`** with `gpt-3.5-turbo`:
     - Include system/user prompts (for direct or CoT).
     - Use `temperature=0.0` for deterministic runs.
     - Seed-setting for reproducibility is recommended.

#### 2. Implement 4 Uncertainty Quantification (UQ) Methods
- Choose 4 methods from Table 2, e.g.:
  - **Log Probability** (token-level confidence).
  - **Confidence Prompt** (model estimates its own certainty).
  - **Self-consistency** (multiple generations → agreement rate).
  - **MC Dropout-like Sampling** (simulate via temperature sampling).
- For each method:
  - Create a function/script (`uq_method_name.py`).
  - Input: Problem text.
  - Output: Answer + confidence score ∈ [0, 1].
  - Save to `results/method_name.json`.

#### 3. Use GPT-3.5 Turbo for All Evaluations
- Optimize API usage:
  - Reuse responses when possible (e.g., answer + confidence in one pass).
  - Batch API calls, handle rate limits.

#### 4. Confidence-Accuracy Curve Evaluation
- Bin results by predicted confidence.
- Compute **accuracy per bin**, then plot:
  - X-axis: Confidence bin (e.g., [0.0–0.2], ... [0.8–1.0]).
  - Y-axis: Accuracy.
- Export:
  - `confidence_vs_accuracy.csv`
  - `confidence_curve.png`
- Optional: Compute **ECE (Expected Calibration Error)**.

---

### 🔹 April 8 – CoT Measurement Setup

#### 1. Define CoT Prompt Templates
- E.g., `"Let's think step by step..."`, `"First, analyze the question..."`.
- Use templates to generate CoT-style prompts.

#### 2. Run CoT Inference
- Use GPT-3.5-Turbo.
- Prompt → full reasoning → final answer.
- Save:
  - Full output (`response['choices'][0]['message']['content']`).
  - Extracted final answer (via regex).
  - Save as `cot_results.json`.

#### 3. Compute CoT Accuracy
- Compare model’s final answer to ground truth.
- Metrics:
  - Accuracy %.
  - Avg. reasoning length.
  - Breakdown of common errors.

#### 4. CoT vs Direct Answering Analysis
- Compare:
  - Final accuracy.
  - Confidence calibration curves.
  - Common failure patterns.

---

### 🔹 April 10–15 – Finalization Tasks

#### April 10:
- Submit:
  - CoT measurement results.
  - Uncertainty results for **grok** and **gemini**.

#### April 13:
- Consolidate:
  - Plots: confidence curve, CoT vs direct comparison.
  - Tables: accuracy by method, confidence calibration.
  - Upload all visuals to slide deck.

#### April 15:
- Presentation:
  - Prepare 1–2 slides covering:
    - UQ methodology.
    - CoT implementation strategy.
    - Key findings and learnings.
    - Example output (good & failed).


