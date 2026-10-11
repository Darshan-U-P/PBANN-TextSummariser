# PBANN-TextSummariser

**An extractive text summarisation system built from scratch using Parallel Brain ANN (PB-ANN).**

PBANN-TextSummariser aims to identify the most important sentences in an article and combine them into a shorter summary while preserving the original wording and sentence order.

The project adapts a PB-ANN neural architecture for sentence-importance prediction, using a trainable text encoder and a recurrent neural core.

## Project Objectives

- Build an extractive text summarisation model from scratch.
- Adapt the PB-ANN architecture for sentence-level importance scoring.
- Develop a trainable text encoder using word embeddings.
- Train the model to select important sentences from articles.
- Evaluate summarisation quality using ROUGE metrics.
- Compare the proposed approach against conventional and neural baselines.

## Architecture

```text
Input Article
     |
     v
Sentence Segmentation
     |
     v
Tokenization & Vocabulary
     |
     v
Trainable Word Embeddings
     |
     v
Sentence Embeddings
     |
     v
PB-ANN Recurrent Neural Core
     |
     v
Sentence Importance Scores
     |
     v
Top-K Sentence Selection
     |
     v
Extractive Summary
```

## Core Components

### 1. Text Encoder

The text encoder converts tokenized sentences into fixed-dimensional sentence representations using trainable word embeddings, masked mean pooling, and a projection network.

Implementation: `src/text_encoder.py`

### 2. PB-ANN Text Scorer

The PB-ANN text scorer adapts a recurrent neural architecture to process sentence representations and predict an importance score for each sentence.

Current components include:

- Trainable input projection
- Recurrent neural weights
- Fixed sparse connectivity mask
- Trainable neuron thresholds
- Leaky integration and threshold-based firing
- Lateral inhibition
- Layer normalization and dropout
- Sentence-importance prediction head

Implementation: `src/pbann.py`

**Implementation note:** The current sparse connectivity mask is applied to a dense weight matrix. Actual sparse-computation speedups have not yet been implemented or established.

### 3. Extractive Summarization

The intended summarization stage will rank sentences by their predicted importance, select the highest-scoring sentences, and restore their original document order.

Implementation: `src/summarizer.py`

## Current Development Status

| Component | Status |
|---|---|
| Project structure | Complete |
| Sentence segmentation and tokenization | Implemented |
| Trainable text encoder | Implemented |
| PB-ANN sentence scorer | Implemented |
| End-to-end gradient flow | Tested |
| Dataset preparation | Planned |
| Sentence-selection label generation | Planned |
| Training pipeline | Planned |
| Inference and summary generation | Planned |
| ROUGE evaluation | Planned |
| Baseline comparisons | Planned |

The current smoke tests verify tensor shapes, forward execution, and gradient flow. They do not yet demonstrate summarisation quality or trained model performance.

## Requirements

- Python 3.12
- PyTorch
- CUDA-compatible NVIDIA GPU recommended
- NLTK

The project has been tested in a CUDA-enabled environment with PyTorch 2.5.1+cu121 on an NVIDIA GeForce RTX 3050 Ti Laptop GPU.

CPU execution is also supported by the current model code, although training performance will differ.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Darshan-U-P/PBANN-TextSummariser.git
cd PBANN-TextSummariser
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Alternatively, activate an existing compatible Python environment.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download NLTK sentence-tokenization resources

```powershell
python -c "import nltk; nltk.download('punkt_tab')"
```

### 5. Run the tests

```powershell
python test_pbann.py
python test_text_encoder.py
python test_text_pipeline.py
```

The end-to-end smoke test checks whether sentence embeddings and importance scores are produced and whether gradients flow through the encoder and PB-ANN recurrent weights.

## Training Plan

The next development stage is to prepare an article–summary dataset and derive approximate sentence-selection targets from reference summaries.

The planned training workflow is:

1. Load and preprocess article–summary examples.
2. Split articles into sentences and tokenize them.
3. Generate sentence-selection targets using a reference-summary matching strategy.
4. Train the text encoder and PB-ANN scorer using a sentence-level classification objective.
5. Validate model performance and tune hyperparameters.
6. Generate extractive summaries from trained sentence scores.
7. Evaluate using ROUGE-1, ROUGE-2, and ROUGE-L.
8. Compare against TF-IDF-based and neural baselines.

Reference-summary matching will produce heuristic labels, which may be noisy. Evaluation results will be needed to determine how effectively the model performs.

## Evaluation

Planned evaluation metrics:

- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest common subsequence overlap

Baseline comparisons and model efficiency measurements will help assess the effectiveness of the PB-ANN approach.

## Repository Structure

```text
PBANN-TextSummariser/
├── data/
├── models/
├── src/
│   ├── pbann.py
│   ├── text_encoder.py
│   └── summarizer.py
├── train.py
├── test_pbann.py
├── test_text_encoder.py
├── test_text_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Research Direction

This project explores whether a PB-ANN architecture can be adapted for extractive text summarisation through sentence-level importance prediction.

Its effectiveness remains an experimental question. Performance claims will be based on trained-model evaluation and comparisons with appropriate baselines.

## License

No license has been selected yet. Until a license is added, reuse and redistribution are subject to the applicable default copyright rules.

## Author

**Darshan U P**

GitHub: [@Darshan-U-P](https://github.com/Darshan-U-P)
