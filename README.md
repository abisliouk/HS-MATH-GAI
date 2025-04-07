# HS-MATH-GAI: Evaluating and Enhancing High School-Level Mathematical Reasoning in LLMs

## Project Summary

This project introduces **HS-MATH-GAI**, a framework to systematically evaluate and improve the **symbolic mathematical reasoning** abilities of Large Language Models (LLMs) on **high school-level math problems**.

The project focuses on three main components:
1. **Benchmark Construction** – Developing a symbolic math benchmark aligned with high school curricula.
2. **Uncertainty Quantification** – Measuring confidence and variability in LLM-generated reasoning chains.
3. **Reasoning Workflow Optimization** – Enhancing reasoning robustness via purification and step-wise refinement techniques.

## Motivation

While LLMs have shown great promise in solving math problems, their **robustness**, **consistency**, and **reliability** in complex symbolic reasoning tasks remain questionable. Most prior benchmarks (e.g., GSM-Symbolic) focus on elementary arithmetic and evaluate outdated models. This project aims to fill this gap by introducing a **high-fidelity testbed and methodology** tailored to modern LLMs and curriculum-relevant challenges.

---

## Project Components

### 1. High School-Level Symbolic Math Benchmark

- Builds on GSM-Symbolic templates but adapts them to high school topics: **algebra**, **functions**, **geometry**, and **probability**.
- Problems will be semantically equivalent but structurally varied to test **reasoning stability**.
- Goals:
  - Assess how model performance varies under **symbolic perturbations**.
  - Evaluate sensitivity to **problem structure** and **complexity**.

### 2. Uncertainty Quantification of Chain-of-Thought Reasoning

- Three comparison conditions:
  - Same model, different random seeds/temperatures.
  - Same model family across capability tiers (e.g., GPT-4 vs GPT-4-Turbo).
  - Different models (e.g., GPT-4, Claude, Gemini).
- Metrics:
  - Token-level entropy
  - Answer agreement
  - Step-level divergence
- Goal: Identify **low-confidence reasoning patterns** and **failure points** in multi-step chains.

### 3. Reasoning Workflow Optimization

- **Problem Purification**: Simplify problem prompts using LLMs to remove distractors while preserving semantics.
- **Chain Refinement**: Detect and regenerate faulty steps in the reasoning chain.
- Objective: Improve **logical consistency** and **final answer accuracy** under uncertainty.

---

## Tools & Infrastructure

- OpenAI API (GPT-3.5, GPT-4, GPT-4-Turbo)
- LLM Evaluation Pipelines for:
  - Prompt generation
  - Sample collection under varied conditions
  - Entropy and agreement analysis
- Dataset augmentation scripts adapted from GSM-Symbolic for symbolic variability

---

## Timeline & Milestones

| Date       | Tasks                                                                 |
|------------|------------------------------------------------------------------------|
| April 6    | API setup, CoT evaluation, problem translation, baseline uncertainty  |
| April 8    | Confidence scoring via purification, baseline evaluation execution    |
| April 10   | Final results for all model tiers and refinement strategy             |
| April 13   | Finalize workflow optimization and slides                             |
| April 15   | Project presentation                                                   |

---

## Team

- **Zhenjiang Mao**
- **Artem Bisliouk**
- **Rohith Reddy Nama**

---

## References

Selected papers used in methodology development:
- Wei et al. (2022) – Chain-of-Thought prompting
- Mirzadeh et al. (2024) – GSM-Symbolic benchmark
- Liu et al. (2023) – Reflexion (Chain refinement)
- Lin et al. (2024) – Prompt agreement and confidence expression
- Portillo Wightman et al. (2023) – Prompt consistency-based uncertainty
