import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary, tokenize
from src.text_encoder import TextEncoder, split_sentences
from src.pbann import PBANNTextScorer


class PBANNSummarizer:
    """
    Extractive text summariser powered by the custom PB-ANN model.

    Pipeline:
        Article
          -> Sentence splitting
          -> Tokenisation and vocabulary encoding
          -> TextEncoder
          -> PB-ANN sentence scoring
          -> Sentence selection
          -> Summary in original sentence order
    """

    def __init__(
        self,
        checkpoint_path=None,
        vocab_path=None,
        threshold=0.60,
        max_sentences=60,
        max_tokens=64,
        max_summary_sentences=3,
        device=None,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
            or ROOT / "models" / "pbann_weighted.pt"
        )

        self.vocab_path = Path(
            vocab_path
            or ROOT / "models" / "vocab.json"
        )

        self.threshold = float(threshold)
        self.max_sentences = int(max_sentences)
        self.max_tokens = int(max_tokens)
        self.max_summary_sentences = int(
            max_summary_sentences
        )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if self.max_sentences < 1:
            raise ValueError(
                "max_sentences must be at least 1."
            )

        if self.max_tokens < 1:
            raise ValueError(
                "max_tokens must be at least 1."
            )

        if self.max_summary_sentences < 1:
            raise ValueError(
                "max_summary_sentences must be at least 1."
            )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable."
            )

        self._validate_files()
        self._load_models()

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    def _validate_files(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        if not self.vocab_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: "
                f"{self.vocab_path}"
            )

    # --------------------------------------------------------
    # Load vocabulary and trained models
    # --------------------------------------------------------

    def _load_models(self):
        print("Loading PB-ANN summariser...")
        print(f"Device: {self.device}")

        # Vocabulary.load is a class method.
        self.vocab = Vocabulary.load(self.vocab_path)
        vocab_size = len(self.vocab)

        # This project uses checkpoints created by train.py.
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )

        required_keys = (
            "encoder_state_dict",
            "scorer_state_dict",
        )

        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(
                    f"Checkpoint is missing '{key}'."
                )

        # Read the actual trained embedding dimensions.
        embedding_weights = checkpoint[
            "encoder_state_dict"
        ]["embedding.weight"]

        expected_vocab_size = embedding_weights.shape[0]
        embedding_dim = embedding_weights.shape[1]

        if vocab_size != expected_vocab_size:
            raise ValueError(
                "Vocabulary/checkpoint mismatch: "
                f"vocabulary has {vocab_size} entries, "
                f"but checkpoint expects "
                f"{expected_vocab_size}."
            )

        num_neurons = checkpoint.get(
            "num_neurons",
            512,
        )

        recurrent_steps = checkpoint.get(
            "recurrent_steps",
            6,
        )

        sparsity = checkpoint.get(
            "sparsity",
            0.8,
        )

        # Recreate the trained encoder.
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        ).to(self.device)

        # Recreate the trained PB-ANN sentence scorer.
        self.scorer = PBANNTextScorer(
            embedding_dim=embedding_dim,
            num_neurons=num_neurons,
            steps=recurrent_steps,
            sparsity=sparsity,
        ).to(self.device)

        self.encoder.load_state_dict(
            checkpoint["encoder_state_dict"]
        )

        self.scorer.load_state_dict(
            checkpoint["scorer_state_dict"]
        )

        self.encoder.eval()
        self.scorer.eval()

        print("Vocabulary loaded:", vocab_size)
        print("Encoder weights loaded.")
        print("PB-ANN weights loaded.")
        print("Summariser ready.")

    # --------------------------------------------------------
    # Split article into sentences
    # --------------------------------------------------------

    def _split_article(self, article):
        sentences = split_sentences(article)

        # Remove empty sentences and normalise whitespace.
        sentences = [
            " ".join(sentence.split())
            for sentence in sentences
            if sentence.strip()
        ]

        # The model was trained with at most 60 sentences.
        return sentences[:self.max_sentences]

    # --------------------------------------------------------
    # Encode article sentences
    # --------------------------------------------------------

    def _encode_sentences(self, sentences):
        encoded_sentences = []

        for sentence in sentences:
            token_ids = self.vocab.encode(
                sentence,
                max_tokens=self.max_tokens,
            )

            encoded_sentences.append(token_ids)

        if not encoded_sentences:
            raise ValueError(
                "No sentences available for encoding."
            )

        sentence_count = len(encoded_sentences)

        longest_sentence = max(
            len(ids) for ids in encoded_sentences
        )

        token_ids = torch.zeros(
            (1, sentence_count, longest_sentence),
            dtype=torch.long,
            device=self.device,
        )

        for sentence_index, ids in enumerate(
            encoded_sentences
        ):
            token_ids[
                0,
                sentence_index,
                :len(ids),
            ] = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            )

        token_mask = token_ids != 0

        sentence_mask = torch.ones(
            (1, sentence_count),
            dtype=torch.bool,
            device=self.device,
        )

        return token_ids, token_mask, sentence_mask

    # --------------------------------------------------------
    # Score article sentences
    # --------------------------------------------------------

    @torch.inference_mode()
    def score_sentences(self, article):
        """
        Return sentence scores without selecting a summary.

        Each result contains:
            index: original sentence index
            sentence: original sentence text
            score: predicted importance probability
        """

        if not isinstance(article, str):
            raise TypeError(
                "article must be a string."
            )

        if not article.strip():
            raise ValueError(
                "Article cannot be empty."
            )

        sentences = self._split_article(article)

        if not sentences:
            raise ValueError(
                "Could not extract any sentences from the article."
            )

        token_ids, token_mask, sentence_mask = (
            self._encode_sentences(sentences)
        )

        sentence_embeddings = self.encoder(
            token_ids,
            token_mask,
        )

        logits = self.scorer(
            sentence_embeddings,
            sentence_mask,
        )

        probabilities = torch.sigmoid(
            logits[0]
        ).cpu().tolist()

        results = []

        for index, (sentence, score) in enumerate(
            zip(sentences, probabilities)
        ):
            results.append({
                "index": index,
                "sentence": sentence,
                "score": float(score),
            })

        return results

    # --------------------------------------------------------
    # Generate extractive summary
    # --------------------------------------------------------

    def summarize(
        self,
        article,
        threshold=None,
        max_summary_sentences=None,
    ):
        """
        Generate an extractive summary.

        Sentences are ranked by predicted importance.
        Selected sentences are returned in their original
        article order.

        If no sentence passes the threshold, the highest-scoring
        sentence is used as a fallback.
        """

        threshold = (
            self.threshold
            if threshold is None
            else float(threshold)
        )

        summary_limit = (
            self.max_summary_sentences
            if max_summary_sentences is None
            else int(max_summary_sentences)
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if summary_limit < 1:
            raise ValueError(
                "max_summary_sentences must be at least 1."
            )

        scored_sentences = self.score_sentences(
            article
        )

        # Select sentences that pass the threshold.
        selected = [
            item
            for item in scored_sentences
            if item["score"] >= threshold
        ]

        # Always allow at least one sentence in the summary.
        if not selected:
            selected = [
                max(
                    scored_sentences,
                    key=lambda item: item["score"],
                )
            ]

        # Limit summary length by selecting the highest scores.
        selected = sorted(
            selected,
            key=lambda item: item["score"],
            reverse=True,
        )[:summary_limit]

        # Restore original article order.
        selected.sort(
            key=lambda item: item["index"]
        )

        return " ".join(
            item["sentence"]
            for item in selected
        )

    # --------------------------------------------------------
    # Summary plus score details
    # --------------------------------------------------------

    def summarize_with_scores(
        self,
        article,
        threshold=None,
        max_summary_sentences=None,
    ):
        """
        Return the summary together with per-sentence scores
        and selection decisions. Useful for debugging.
        """

        threshold = (
            self.threshold
            if threshold is None
            else float(threshold)
        )

        summary_limit = (
            self.max_summary_sentences
            if max_summary_sentences is None
            else int(max_summary_sentences)
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if summary_limit < 1:
            raise ValueError(
                "max_summary_sentences must be at least 1."
            )

        scored = self.score_sentences(article)

        selected = [
            item for item in scored
            if item["score"] >= threshold
        ]

        if not selected:
            selected = [
                max(scored, key=lambda item: item["score"])
            ]

        selected = sorted(
            selected,
            key=lambda item: item["score"],
            reverse=True,
        )[:summary_limit]

        selected_indices = {
            item["index"] for item in selected
        }

        # Mark selected sentences and preserve article order.
        sentence_details = []

        for item in scored:
            sentence_details.append({
                **item,
                "selected": (
                    item["index"] in selected_indices
                ),
            })

        selected.sort(
            key=lambda item: item["index"]
        )

        summary = " ".join(
            item["sentence"]
            for item in selected
        )

        return {
            "summary": summary,
            "threshold": threshold,
            "sentences": sentence_details,
        }


# ============================================================
# Simple command-line demonstration
# ============================================================

if __name__ == "__main__":
    summarizer = PBANNSummarizer()

    print("\nEnter an article below.")
    print("Enter an empty line to finish.\n")

    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line.strip():
            break

        lines.append(line)

    article = "\n".join(lines)

    if not article.strip():
        print("No article provided.")
        raise SystemExit(0)

    try:
        result = summarizer.summarize_with_scores(
            article,
            threshold=0.60,
            max_summary_sentences=3,
        )

        print("\n" + "=" * 68)
        print("EXTRACTIVE SUMMARY")
        print("=" * 68)
        print(result["summary"])

        print("\n" + "=" * 68)
        print("SENTENCE IMPORTANCE SCORES")
        print("=" * 68)

        for item in result["sentences"]:
            status = (
                "SELECTED"
                if item["selected"]
                else "not selected"
            )

            print(
                f"[{item['score']:.4f}] "
                f"{status}: {item['sentence']}"
            )

    except Exception as exc:
        print(f"\nSummarisation failed: {exc}")
        raise
