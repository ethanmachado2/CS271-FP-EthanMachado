# CS271 Final Project — FOMC Stance Classification & Market Prediction

**Ethan Machado | CS 271 — Natural Language Processing | San José State University**

---

## Research Questions

**RQ1:** How do domain-specific, fine-tuned transformer models perform against rule-based approaches and generic LLMs in classifying the policy stance of FOMC sentences using domain-specific labels (dovish / hawkish / neutral)?

**RQ2:** Is there an observed improvement in short-term (3-day) S&P 500 directional movement prediction when including a domain-specific LLM stance feature compared to a traditional sentiment feature?

---

## Overview

This project builds a full NLP pipeline over Federal Open Market Committee (FOMC) meeting text spanning 2006–2024. The pipeline covers:

- **Phase 1** — Data acquisition: FOMC sentence corpora, S&P 500 returns (yfinance), FRED macro series (CPI, unemployment)
- **Phase 2** — Definitive data splits and bag-of-words baselines (TF-IDF + Logistic Regression, TF-IDF + SVM)
- **Phase 3** — BERT-family grid search (bert-base-uncased, ProsusAI/finbert, distilbert-base-uncased) and zero-shot evaluation (ZiweiChen/FinBERT-FOMC); few-shot LLM labeling via Groq and AWS Bedrock (Claude 3.5, Llama 3.3, GPT-OSS)
- **Phase 4** — Event-level RQ2 pipeline: sentence → FOMC-meeting aggregation, macro controls, Logistic Regression and Random Forest classifiers, robustness checks, error analysis, inference benchmarking, and final results with 95% bootstrap CIs

All experiments are tracked in MLflow (`CS271-RQ1-Stance` and `CS271-RQ2-Market` experiments, SQLite backend).

---

## Results Summary

### RQ1 — Stance Classification (test n = 194, macro-F1)

| Model | Accuracy | Macro F1 | 95% CI |
|---|---|---|---|
| Claude 3.5 Sonnet few-shot | 0.6598 | 0.6503 | — |
| Claude 3 Opus few-shot | 0.6546 | 0.6455 | — |
| Llama 3.3 70B (BR) few-shot | 0.6340 | 0.6310 | — |
| **bert-base-uncased fine-tuned** | 0.6186 | 0.6223 | [0.269, 0.400] |
| GPT-OSS 120B (BR) few-shot | 0.6134 | 0.6126 | — |
| GPT-OSS 20B few-shot | 0.5979 | 0.6012 | — |
| distilbert-base-uncased fine-tuned | 0.5670 | 0.5415 | [0.257, 0.395] |
| TF-IDF + Logistic Regression | 0.5567 | 0.5233 | [0.256, 0.385] |
| ProsusAI/finbert fine-tuned | 0.5206 | 0.5052 | [0.262, 0.396] |
| TF-IDF + SVM | 0.5103 | 0.5011 | [0.265, 0.396] |
| FinBERT-FOMC zero-shot | 0.4588 | 0.4506 | [0.257, 0.386] |

Key finding: LLMs with few-shot prompting outperform all fine-tuned transformers. Among fine-tuned models, the generic bert-base-uncased outperforms the domain-specific FinBERT, suggesting the small training set (≈580 sentences) limits domain adaptation.

### RQ2 — 3-Day S&P 500 Direction (temporal test set)

| Model | Accuracy | 95% CI |
|---|---|---|
| Majority class baseline | 0.6667 | [0.481, 0.815] |
| Macro + Generic Sentiment | 0.6296 | [0.296, 0.667] |
| RF (Macro + Sent + LLM Stance) | 0.6296 | [0.444, 0.815] |
| Macro + LLM Stance | 0.5926 | [0.296, 0.667] |
| Macro + Sent + LLM Stance (LR) | 0.5926 | [0.296, 0.667] |

Key finding: No feature set reliably beats the majority class baseline within bootstrap CI bounds, consistent with the efficient market hypothesis. All CIs overlap substantially, indicating that with only ~27 temporal test events, the sample is too small to detect a statistically significant edge from stance features.

### Inference Efficiency (CPU, n = 194 sentences)

| Model | Total (ms) | Per-sentence |
|---|---|---|
| Event-level LR (RQ2) | 0.02 | 0.9 µs |
| TF-IDF + Logistic Regression | 2.6 | 13.5 µs |
| TF-IDF + SVM | 16.2 | 83.5 µs |
| FinBERT-FOMC zero-shot (CPU) | 3 087.9 | 15 917 µs |
| LLM via Groq API | ~40–80 ms/sentence | network-bound |
| LLM via AWS Bedrock | ~50–150 ms/sentence | network-bound |

---

## Repository Structure

```
Phases 3 and 4/
├── CS271_FinalProject_EM_Bedrock.ipynb   # Main notebook (all phases)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── FOMC.csv                      # Dated FOMC sentences (RQ2 corpus)
│   │   └── manual-mm-split.csv           # Manually labeled stance corpus (RQ1)
│   └── processed/
│       ├── fomc_market_master.csv         # Merged FOMC + S&P 500 master dataset
│       ├── fomc_labeled_df.csv
│       ├── split_train/val/test.csv       # RQ1 definitive splits
│       ├── event_df.csv                   # Event-level features (stance + macro)
│       ├── event_train/test.csv           # RQ2 temporal splits
│       ├── master_ev_with_stance.csv
│       ├── macro_cpi.csv / macro_unrate.csv
│       ├── rq2_results_3d.csv
│       ├── rq2_results_final_with_ci.csv  # Final RQ2 table with bootstrap CIs
│       ├── robustness_a_horizon.csv       # 10-day horizon robustness
│       ├── robustness_b_subperiod.csv     # Temporal sub-period robustness
│       └── robustness_c_truncation.csv    # Input truncation robustness
│
├── bert-uncased_results/                  # bert-base-uncased grid search outputs
├── finbert_results/                       # ProsusAI/finbert grid search outputs
├── distillbert_results/                   # distilbert-base-uncased grid search outputs
├── finbert-fomc_results/                  # FinBERT-FOMC zero-shot predictions
│
├── bert-uncased_runs/                     # HuggingFace Trainer checkpoints
├── finbert_runs/
├── distilbert_runs/
│
├── llm_outputs/                           # LLM labeling checkpoints (per model/corpus)
│   ├── claude46-sonnet/
│   ├── claude46-opus/
│   ├── llama33-70b-br/
│   ├── gpt-oss-120b-br/
│   └── ...
│
├── mlflow.db                              # MLflow SQLite tracking store
└── mlruns/                                # MLflow artifact store
```

---

## Setup

### Requirements

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

### API Credentials

The notebook uses two external LLM backends. Set the following environment variables before running the labeling cells:

| Backend | Variable | Where to obtain |
|---|---|---|
| Groq | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| AWS Bedrock | AWS credentials via `~/.aws/credentials` or environment | AWS IAM + Bedrock model access enabled in console |

LLM labeling is disabled by default (`RUN_LLM_LABELING = False`, `RUN_BEDROCK_LABELING = False`). Pre-computed labels are loaded from `llm_outputs/` automatically.

---

## Running the Notebook

1. Open `CS271_FinalProject_EM_Bedrock.ipynb` in VS Code or JupyterLab.
2. Run **cell 2** first (imports + CWD normalization).
3. Run cells in order. The notebook is designed to be fully reproducible from pre-computed results — no re-training or re-labeling is required by default.

**To reproduce BERT training** from scratch: set `RUN_BERT_TRAINING = True` in the Phase 3 config cell. Expect several hours on CPU; GPU strongly recommended.

**To reproduce LLM labeling**: set `RUN_LLM_LABELING = True` (Groq) or `RUN_BEDROCK_LABELING = True` (AWS Bedrock) and ensure the corresponding credentials are set. Labels are checkpointed per model so runs are safely resumable.

---

## MLflow Experiment Tracking

Two MLflow experiments are logged to `mlflow.db` (SQLite):

| Experiment | Contents |
|---|---|
| `CS271-RQ1-Stance` | TF-IDF baselines (1 run each); BERT grid search (24 runs per fine-tuned model); FinBERT-FOMC zero-shot (1 run) |
| `CS271-RQ2-Market` | Majority class and VADER baselines; all event-level LR/RF feature-set comparisons |

To browse results locally:

```bash
cd "Phases 3 and 4"
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Key Design Decisions

- **Single split definition** — RQ1 and RQ2 splits are defined once in Phase 2 and persisted to `data/processed/`. No model sees the test set during training or hyperparameter selection.
- **Chronological RQ2 split** — FOMC events are split 80/20 by meeting date (no shuffling) to prevent look-ahead bias.
- **Class-balanced loss** — All classifiers use class-weighted cross-entropy or `class_weight='balanced'` to account for the neutral-heavy label distribution.
- **Bootstrap CIs** — Final metric comparisons use 2,000-resample bootstrap CIs to avoid over-interpreting small test set differences.
- **Unified LLM backend** — Groq and AWS Bedrock share the same 7-shot prompt, batch size, and checkpointing logic, making results directly comparable across providers.

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| `FOMC.csv` | Prior research corpus | 908 FOMC sentences with GPT-4 growth/inflation/employment stance labels and meeting dates |
| `manual-mm-split.csv` | Manual annotation | 966 FOMC sentences with hawkish/dovish/neutral stance labels |
| S&P 500 prices | yfinance (SPY ETF) | Daily OHLCV, 2006–2025; 3-day and 10-day forward returns computed in-notebook |
| FRED macro series | FRED public API | CPI (CPIAUCSL) and unemployment rate (UNRATE), monthly, merged to nearest FOMC event date |
