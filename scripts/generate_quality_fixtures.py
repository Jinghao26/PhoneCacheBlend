#!/usr/bin/env python3
"""Export quality suite JSON fixtures for CacheBlend reference runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from quality_suite_data import (  # noqa: E402
    PASS_THRESHOLD,
    PASSAGES,
    SUITES,
    format_passage_chunk,
    passage_texts,
)


def build_suite_payload(suite_key: str) -> dict:
    suite = SUITES[suite_key]
    queries = []
    for q in suite["queries"]:
        passages = passage_texts(q["passage_ids"])
        chunk_texts = [format_passage_chunk(i, p) for i, p in enumerate(passages)]
        queries.append(
            {
                "id": q["id"],
                "title": q["title"],
                "passage_ids": q["passage_ids"],
                "passages": passages,
                "chunk_texts": chunk_texts,
                "question": q["question"],
                "key_phrases": q["key_phrases"],
                "max_tokens": q.get("max_tokens", suite["default_max_tokens"]),
            }
        )
    return {
        "suite": suite_key,
        "name": suite["name"],
        "pass_threshold": PASS_THRESHOLD,
        "default_max_tokens": suite["default_max_tokens"],
        "passages": {str(k): v for k, v in PASSAGES.items()},
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "CacheBlend" / "inputs",
        help="Directory for quality_simple.json and quality_harder.json",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for suite_key in ("simple", "harder"):
        payload = build_suite_payload(suite_key)
        out_path = args.out_dir / f"quality_{suite_key}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path} ({len(payload['queries'])} queries)")


if __name__ == "__main__":
    main()
