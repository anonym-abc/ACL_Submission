# Auditing Numeric Outputs of Large Language Models via Air-Quality Queries

The repository supports reproducibility of all experiments reported in the main paper and supplementary material, including zero-shot evaluation of large language models (LLMs), fine-tuning experiments, supervised machine-learning baselines, and bias analyses across geographic, socioeconomic, and digital-visibility dimensions.

---

## Repository Structure

```
LLM_Bias/
├── Code/
│   ├── Analysis/
│   ├── Fine_Tuning/
│   ├── ML_Baseline/
│   └── Zero_Shot/
└── Dataset/
```
---

## Code

### Analysis

This directory contains all analysis and visualization notebooks used to generate figures and tables in the main paper and supplementary material.

Bias analyses in this directory are performed only for zero-shot LLM predictions. All analyses related to fine-tuned models are consolidated in a single notebook.

Zero-shot bias analysis notebooks:
- Media_bias.ipynb  
  Digital visibility bias analysis comparing cities with and without news coverage.

- Regional_bias.ipynb  
  Geographic bias analysis across Indian regions (North, South, East, West, Central, Northeast).

- Socioeconomic_bias.ipynb  
  Bias analysis across administrative city tiers and multidimensional poverty index (MPI) groups.

- Seasonal_temperature_prompt_variations.ipynb  
  Robustness analysis covering prompt phrasing variations, decoding temperature sensitivity, and prediction stability.

Fine-tuned model analysis:
- FineTune_Analysis.ipynb  
  Contains all evaluation and comparison results for fine-tuned LLMs, including comparisons with zero-shot LLMs and supervised baselines.

---

### Zero_Shot

This directory contains example implementations for zero-shot LLM inference.

- GPT-4o.ipynb  
  Example notebook demonstrating zero-shot PM2.5 estimation using a closed-source LLM.

- gemma9b.py  
  Example Python script for zero-shot inference using Gemma-2-9B-it.

Only one representative zero-shot model is included as a sample. The full study evaluates multiple models following the same querying protocol. Depending on model interfaces, some implementations use separate system and user prompts, while others use a combined prompt.

---

### Fine_Tuning

This directory contains example fine-tuning code for open-weight LLMs.

- gemma2_9b.ipynb  
  Example notebook demonstrating parameter-efficient fine-tuning for Gemma-2-9B-it using 2022 CPCB data.

Only one fine-tuned model is included as a reference implementation. All fine-tuned results reported in the paper follow the same training and evaluation protocol.

---

### ML_Baseline

This directory contains supervised machine-learning baselines used for comparison with LLMs.

- linear_regression.ipynb  
- Random_Forest.ipynb  
- XGBoost.ipynb  
- simple_heuristic.ipynb  

All baselines use minimal geographic and temporal features and are trained on 2022 data and evaluated on 2023 data.

---

## Dataset

This directory contains processed datasets used in the experiments.

Key files include:
- Ground_Truth_2023_Final.csv  
  Monthly city-level PM2.5 ground truth derived from CPCB monitoring stations.

- train_2022.csv, test_2023.csv  
  Train-test splits used for fine-tuning and supervised baselines.

- Wiki_News_Data.csv  
  City-level news article counts used as a proxy for digital visibility.

- mpi_2023.csv  
  Multidimensional Poverty Index (MPI) values used for socioeconomic stratification.

- region.txt, city_tire.csv, city_monthly_ndvi_2023.csv  
  Auxiliary metadata used in bias and robustness analyses.

All datasets are preprocessed and ready for direct use with the provided notebooks.

---

## Prompting Protocol

All LLMs are queried using a text-to-number formulation.

System prompt:
You are an air pollution assistant. Strictly respond to queries with a single real number only. Do not include any units, explanation, or punctuation.

User prompt (base version):
What is the average PM2.5 concentration (in μg/m³) in {city}, {state} during {month}, {year}? Give a single number only.

Additional prompt variants are evaluated for robustness and documented in the analysis notebooks.

---

## Reproducibility Notes

All experiments are conducted using the Hugging Face ecosystem. Deterministic decoding (temperature = 0) is used unless stated otherwise. Experiments were run on NVIDIA A100 (80GB) GPUs. API tokens and credentials are not included and must be provided via environment variables.
