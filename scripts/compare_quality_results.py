#!/usr/bin/env python3
"""
Compare CacheBlend reference results vs PhoneCacheBlend (llama.cpp) results.

PhoneCacheBlend JSON can be produced manually or via the template below after running
Quality Test in the iOS app.

Example:
  python scripts/compare_quality_results.py \\
    --reference ../CacheBlend/results/cacheblend_simple.json \\
    --phone results/phone_simple.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def fmt_ms(x: float) -> str:
    return f"{x:.0f} ms"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = " | "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))


def compare(payload_ref: dict, payload_phone: dict) -> None:
    ref_runs = {r["id"]: r for r in payload_ref["runs"]}
    phone_runs = {r["id"]: r for r in payload_phone["runs"]}
    common_ids = sorted(set(ref_runs) & set(phone_runs))
    if not common_ids:
        print("No overlapping query IDs between reference and phone results.", file=sys.stderr)
        sys.exit(1)

    print("Platform comparison")
    print(f"  Reference: {payload_ref.get('platform')} / {payload_ref.get('model')}")
    print(f"  Phone:     {payload_phone.get('platform')} / {payload_phone.get('model')}")
    print(f"  Suite:     {payload_ref.get('suite_name', payload_ref.get('suite'))}")
    print()

    rows: list[list[str]] = []
    for qid in common_ids:
        ref = ref_runs[qid]
        phone = phone_runs[qid]

        ref_cb = ref["cacheblend"]
        ref_bl = ref["baseline"]
        phone_cb = phone.get("phonecacheblend") or phone.get("cacheblend")
        phone_bl = phone.get("baseline")

        if phone_cb is None or phone_bl is None:
            print(f"Skipping Q{qid}: phone result missing cacheblend/baseline keys", file=sys.stderr)
            continue

        rows.append(
            [
                str(qid),
                ref.get("title", phone.get("title", ""))[:28],
                f"{fmt_pct(ref_bl['score'])} / {fmt_pct(ref_cb['score'])}",
                f"{fmt_pct(phone_bl['score'])} / {fmt_pct(phone_cb['score'])}",
                f"{fmt_ms(ref_bl['ttft_ms'])} / {fmt_ms(ref_cb['ttft_ms'])}",
                f"{fmt_ms(phone_bl['ttft_ms'])} / {fmt_ms(phone_cb['ttft_ms'])}",
            ]
        )

    print_table(
        ["Q", "Title", "Ref BL/CB score", "Phone BL/CB score", "Ref BL/CB TTFT", "Phone BL/CB TTFT"],
        rows,
    )

    print()
    for label, key in (("Reference CacheBlend", "cacheblend"), ("Reference baseline", "baseline")):
        s = payload_ref["summary"][key]
        print(
            f"{label}: {s['pass_count']}/{s['query_count']} pass, "
            f"avg score {fmt_pct(s['avg_score'])}, avg TTFT {fmt_ms(s['avg_ttft_ms'])}"
        )
    for label, key in (("PhoneCacheBlend", "phonecacheblend"), ("Phone baseline", "baseline")):
        if key not in payload_phone["summary"] and key == "phonecacheblend":
            key = "cacheblend"
        s = payload_phone["summary"][key]
        print(
            f"{label}: {s['pass_count']}/{s['query_count']} pass, "
            f"avg score {fmt_pct(s['avg_score'])}, avg TTFT {fmt_ms(s['avg_ttft_ms'])}"
        )

    print()
    print("Notes:")
    print("- Reference uses Mistral-7B on CUDA; Phone uses Qwen2.5-1.5B on device — absolute scores/TTFT are not directly comparable.")
    print("- Compare relative patterns: does CacheBlend preserve quality vs baseline on both platforms?")
    print("- Compare TTFT ratio (baseline/cacheblend) on each platform separately.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--phone", type=Path, required=True)
    args = parser.parse_args()
    compare(load(args.reference), load(args.phone))


if __name__ == "__main__":
    main()
