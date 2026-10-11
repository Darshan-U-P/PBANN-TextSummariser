from __future__ import annotations

import re
from pathlib import Path

import trafilatura

ROOT = Path(__file__).resolve().parents[1]
URL = "https://icymi.in/article/methods-and-goals-in-ai"
OUTPUT_PATH = ROOT / "data" / "real_article.txt"


def clean_extracted_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"^\s*\d{1,2},\s+[A-Za-z]+\s+\d{4},\s*"
        r"\d{1,2}:\d{2}\s*(?:AM|PM)\s*(?:ET|EST|EDT|UTC)?\s*"
        r"\(AP\)\s*\.{0,3}\s*(?:\.\.\.\s*)?\(?Show more\)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def main() -> None:
    downloaded = trafilatura.fetch_url(URL)
    if not downloaded:
        raise RuntimeError(
            f"Could not download {URL}. Check your internet connection "
            "or save the article text manually to data/real_article.txt."
        )

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text or len(text.split()) < 100:
        raise RuntimeError(
            "Extraction returned fewer than 100 words. Inspect the source page "
            "or save the article text manually; do not evaluate an incomplete page."
        )

    text = clean_extracted_text(text)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text + "\n", encoding="utf-8")

    print(f"Saved cleaned article to: {OUTPUT_PATH}")
    print(f"Extracted words: {len(text.split())}")
    print("Preview:")
    print(text[:350])


if __name__ == "__main__":
    main()
