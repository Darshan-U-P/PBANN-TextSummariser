# PBANN-TextSummariser

**An extractive text summarisation system built from scratch using Parallel Brain ANN (PB-ANN).**

PBANN-TextSummariser identifies important sentences in an article and combines them into a shorter summary while preserving the original wording and relative sentence order.

The project explores adapting a PB-ANN neural architecture for sentence-importance prediction using trainable text representations and a recurrent neural scoring component. It evaluates the resulting extractive summariser against conventional baselines, including Lead-3, using ROUGE metrics.

> **Current status:** The training pipeline, trained checkpoints, inference workflow, and held-out evaluation are operational. Initial results show that PBANN is competitive with Lead-3 on some metrics, but a consistent overall improvement has not been demonstrated.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Dataset and Training](#dataset-and-training)
- [Evaluation Results](#evaluation-results)
- [Real-Article Inference Test](#real-article-inference-test)
- [Limitations](#limitations)
- [Development Status](#development-status)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running Inference and Evaluation](#running-inference-and-evaluation)
- [Repository Structure](#repository-structure)
- [Research Direction](#research-direction)
- [License](#license)

## Project Objectives

- Build an extractive text summarisation model from scratch.
- Adapt the PB-ANN architecture for sentence-level importance scoring.
- Develop trainable text representations using word embeddings.
- Train the model to identify important sentences in articles.
- Generate summaries by ranking and selecting source sentences.
- Evaluate summary quality using ROUGE-1, ROUGE-2, and ROUGE-L.
- Compare the proposed approach against conventional baselines.
- Investigate the effectiveness and limitations of the PB-ANN approach for extractive summarisation.

## Architecture

```text
Input Article
     |
     v
Text Cleaning and Sentence Segmentation
     |
     v
Tokenization and Vocabulary Lookup
     |
     v
Trainable Word Embeddings
     |
     v
Sentence Representation / Encoder
     |
     v
PB-ANN Sentence-Scoring Network
     |
     v
Sentence Importance Scores
     |
     v
Top-K Sentence Selection
     |
     v
Restore Original Sentence Order
     |
     v
Extractive Summary
```

The model predicts sentence-importance scores. The summarisation pipeline selects the highest-scoring sentences and restores their original document order to produce a readable extractive summary.

## Core Components

### 1. Text Encoder

The text encoder converts tokenized sentences into fixed-dimensional sentence representations.

The implemented design uses:

- Trainable word embeddings.
- Masked mean pooling.
- A projection network for sentence-level representations.

Implementation: `src/text_encoder.py`

### 2. PB-ANN Text Scorer

The PB-ANN scorer processes sentence representations and predicts an importance score for each sentence.

The architecture includes:

- Trainable input projection.
- Recurrent neural weights.
- Fixed sparse connectivity masks.
- Trainable neuron thresholds.
- Leaky integration and threshold-based firing.
- Lateral inhibition.
- Layer normalization and dropout.
- A sentence-importance prediction head.

Implementation: `src/pbann.py`

**Implementation note:** The sparse connectivity mask is applied to a dense weight matrix. Dedicated sparse computation and its performance benefits have not yet been established.

### 3. Extractive Summarisation

The inference pipeline:

1. Cleans and segments an input article.
2. Converts sentences into model-compatible representations.
3. Generates sentence-importance scores.
4. Ranks sentences by score.
5. Selects the requested number of sentences.
6. Restores their original source order.
7. Returns the extractive summary and sentence-ranking information.

The existing inference and validation scripts use trained model checkpoints and the project vocabulary.

### 4. Evaluation Pipeline

The evaluation workflow calculates ROUGE metrics on a held-out test set and compares PBANN against Lead-3.

Implementations and reports are maintained under `scripts/` and `evaluation_results/`.

## Dataset and Training

The prepared dataset contains article–summary examples split into three files.

| Split | Examples |
|---|---:|
| Training | 10,000 |
| Validation | 500 |
| Test | 500 |
| **Total** | **11,000** |

Dataset files:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`

The vocabulary is stored in `models/vocab.json` and contains 50,000 entries.

### Training methodology

The training workflow uses reference summaries to derive approximate sentence-selection labels. These labels are generated through a greedy ROUGE-based sentence-matching strategy.

The model is trained to predict sentence-level importance scores using these derived targets.

**Important:** The generated labels are heuristic targets rather than manually annotated sentence-importance labels. Their quality may affect model performance.

Three model seeds were evaluated:

- Seed 42
- Seed 123
- Seed 2026

The corresponding position-aware checkpoints are stored under `models/multiseed/`.

## Evaluation Results

### Held-out test-set performance

The following results were obtained on 500 held-out test examples.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| PBANN — Seed 42 | 0.3031 | 0.1180 | 0.2078 |
| PBANN — Seed 123 | 0.3040 | 0.1182 | 0.2096 |
| PBANN — Seed 2026 | 0.3047 | 0.1215 | 0.2106 |
| **PBANN — Mean** | **0.3040** | **0.1192** | **0.2094** |
| Lead-3 baseline | 0.3027 | 0.1194 | 0.2068 |
| **PBANN minus Lead-3** | **+0.0013** | **−0.0002** | **+0.0026** |

*Scores are rounded to four decimal places. Differences are calculated from the displayed mean and baseline values.*

### Interpretation

The results show a mixed outcome:

- **ROUGE-1:** PBANN achieves a small improvement over Lead-3.
- **ROUGE-2:** PBANN is marginally below Lead-3.
- **ROUGE-L:** PBANN achieves a small improvement over Lead-3.
- **Across random seeds:** Results are reasonably close, although the three runs do not establish statistical significance.

The mean PBANN results are slightly better on ROUGE-1 and ROUGE-L, but not ROUGE-2. Therefore, the current experiment does not demonstrate a consistent overall advantage over Lead-3.

These results establish a useful experimental baseline for further development, rather than evidence that PB-ANN is superior to conventional summarisation methods.

### Earlier validation results

The validation experiments also examined a position-aware configuration.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Position-aware PBANN — mean | 0.2887 | 0.1088 | 0.1955 |
| Lead-3 baseline | 0.2911 | 0.1112 | 0.1949 |

On validation data, Lead-3 performed better on ROUGE-1 and ROUGE-2, while the position-aware model was marginally better on ROUGE-L.

The available bootstrap confidence intervals for the validation comparison included zero. The observed differences therefore do not establish a statistically reliable advantage.

### Evaluation artifacts

The final evaluation reports are saved in:

- `evaluation_results/final_validation/final_test_report.json`
- `evaluation_results/final_validation/final_test_report.md`

These reports provide the basis for the reported held-out evaluation results.

## Real-Article Inference Test

The project has also been tested on an independently supplied article stored in:

`data/real_article.txt`

The inference script reads the article from disk, processes its sentences, loads the trained seed-42 checkpoint, scores the sentences, selects the highest-scoring sentences, and generates an extractive summary.

Script: `scripts/test_real_article.py`

### Initial observed results

| Measurement | Result |
|---|---:|
| Sentences extracted | 62 |
| Sentences evaluated | 60 |
| Input words evaluated | 1,044 |
| Summary sentences | 3 |
| Summary words | 64 |
| Compression ratio | 93.87% |

The generated summary selected three sentences from the article, including the opening discussion of artificial intelligence and a sentence explaining how AI works.

The initial run demonstrated that the model could load a checkpoint and produce a summary from a real text file.

### Interpretation and limitations

The first run also exposed preprocessing issues:

- The title and introduction were combined into a single sentence.
- Some list formatting and section headings were not handled cleanly.
- The inference script evaluated only the first 60 sentences.

Consequently, the real-article test is a functional demonstration, not a formal quality evaluation. The 93.87% compression ratio describes the reduction in word count; it does not measure summary accuracy or usefulness.

The initial real-article result is saved to:

`evaluation_results/final_validation/real_article_result.json`

The preprocessing workflow is being improved before conducting broader real-world testing.

## Limitations

The current implementation has several limitations.

1. **Heuristic training labels:** Sentence-selection targets are derived from reference-summary overlap rather than human annotation.
2. **Mixed baseline performance:** PBANN does not consistently outperform Lead-3 across all three ROUGE metrics.
3. **Limited evaluation scope:** The formal held-out evaluation contains 500 examples from the prepared dataset.
4. **Preprocessing sensitivity:** Headings, numbered lists, extraction artefacts, and sentence boundaries can influence inference quality.
5. **Sparse-computation overhead:** Sparse connectivity masks are applied to dense matrices, so actual sparse-computation speedups have not been demonstrated.
6. **Extractive-only output:** The system selects existing sentences; it does not generate paraphrases or synthesise new sentences.
7. **Generalisation remains uncertain:** Broader evaluation across different domains and document types is needed.

## Development Status

| Component | Status |
|---|---|
| Project structure and core architecture | Implemented |
| Trainable text encoder | Implemented |
| PB-ANN sentence scorer | Implemented |
| Forward execution and gradient-flow tests | Tested |
| Dataset preparation | Completed for current experiments |
| Heuristic sentence-selection labels | Implemented |
| Training pipeline | Operational |
| Trained model checkpoints | Available |
| Inference and extractive summary generation | Operational |
| ROUGE evaluation | Completed for current test set |
| Lead-3 baseline comparison | Completed |
| Multi-seed evaluation | Completed for three seeds |
| Real-article inference | Tested; preprocessing improvements needed |
| Frontend application | Planned |
| Backend API | Planned |
| Integrated evaluation dashboard | Planned |
| Dedicated sparse computation | Not yet implemented or benchmarked |

## Requirements

The existing ML environment has been tested with:

- Python 3.12.
- PyTorch 2.5.1+cu121.
- CUDA-compatible NVIDIA GPU.
- NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB VRAM.
- NLTK for sentence tokenization.

CPU inference is also supported by the current model code, although performance will differ.

The precise dependency versions should be taken from the repository's current `requirements.txt` rather than assumed from this list.

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/Darshan-U-P/PBANN-TextSummariser.git
cd PBANN-TextSummariser
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Alternatively, activate an existing compatible environment.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with your NVIDIA driver and CUDA runtime.

### 4. Download NLTK tokenizer resources

```powershell
python -c "import nltk; nltk.download('punkt_tab')"
```

### 5. Run the core tests

```powershell
python test_pbann.py
python test_text_encoder.py
python test_text_pipeline.py
```

Run these tests from the repository root.

## Running Inference and Evaluation

### Real-article inference

Ensure `data/real_article.txt` contains the article text, then run:

```powershell
python .\scripts\test_real_article.py
```

The script prints sentence rankings, the PBANN summary, a Lead-3 summary, structural checks, and the path to the JSON report.

### Final validation

The final-validation scripts and saved reports are located under:

```text
scripts/final_validation.py
evaluation_results/final_validation/
```

Use the repository's existing validation workflow to reproduce the held-out evaluation. Do not treat a single real-article example as a substitute for the formal test set.

## Repository Structure

```text
PBANN-TextSummariser/
├── data/
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── real_article.txt
├── models/
│   ├── vocab.json
│   └── multiseed/
│       ├── position_seed_42_best.pt
│       ├── position_seed_123_best.pt
│       └── position_seed_2026_best.pt
├── src/
│   ├── pbann.py
│   ├── text_encoder.py
│   ├── dataset.py
│   └── summarizer.py
├── scripts/
│   ├── train_position_multiseed.py
│   ├── final_validation.py
│   └── test_real_article.py
├── evaluation_results/
│   └── final_validation/
│       ├── final_test_report.json
│       ├── final_test_report.md
│       └── real_article_result.json
├── test_pbann.py
├── test_text_encoder.py
├── test_text_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

The exact repository contents may evolve as the frontend and backend are added.

## Future Development

The next development stage is to turn the research implementation into a complete application.

Planned work includes:

1. Improve sentence segmentation, heading detection, and document preprocessing.
2. Build a FastAPI backend around the existing inference pipeline.
3. Develop a React frontend for article input and summary display.
4. Display sentence rankings and compare PBANN against Lead-3.
5. Add automated API and integration tests.
6. Evaluate on more documents and report confidence intervals where appropriate.
7. Investigate better training labels and alternative sentence-ranking objectives.
8. Benchmark latency, memory use, and actual sparse-computation implementations.

## Research Direction

This project investigates whether a PB-ANN architecture can be adapted for extractive text summarisation through sentence-level importance prediction.

The current implementation demonstrates an operational training and inference pipeline. Held-out evaluation shows mixed results relative to Lead-3, leaving the effectiveness of the architecture as an open empirical question.

Future claims about accuracy, generalisation, efficiency, or superiority will be based on reproducible experiments and appropriate baselines.

## License

No license has been selected yet. Until a license is added, reuse and redistribution are subject to applicable copyright rules.

## Author

**Darshan U P**

GitHub: [@Darshan-U-P](https://github.com/Darshan-U-P)
