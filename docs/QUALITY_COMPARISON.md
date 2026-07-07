# Quality Test Cross-Platform Comparison

Compare **original CacheBlend** (vLLM + CUDA) against **PhoneCacheBlend** (llama.cpp on iPhone) using the same `Simple_test` and `Harder_test` fixtures.

## What is being compared

| Dimension | Original CacheBlend | PhoneCacheBlend |
|-----------|---------------------|-----------------|
| Code | `CacheBlend/vllm_blend` | `PhoneCacheBlend/llama.cpp` |
| Model | Mistral-7B-Instruct (required — fuse hooks in `llama.py`) | Qwen2.5-1.5B-Instruct Q4_K_M |
| Hardware | NVIDIA GPU (≥40 GB VRAM recommended) | iPhone / simulator |
| Baseline | Full prefill (`check=False`) | Standard llama (full prefill) |
| Fuse path | `check=True`, `recomp_ratio=0.18` | GRAPH fuse, `hkvdRecompRatio=0.18` |
| Quality metric | Key-phrase coverage ≥70% (same as app) | Key-phrase coverage ≥70% |

**Important:** Absolute TTFT and answer quality are **not** apples-to-apples across platforms (different model size and hardware). The useful comparison is:

1. **Quality delta:** Does fuse/CacheBlend lose fewer key phrases than baseline on each platform?
2. **Speedup ratio:** `baseline_TTFT / cacheblend_TTFT` on each platform separately.
3. **Failure patterns:** Which query IDs fail on both vs only on PhoneCacheBlend?

## Step 1 — Generate shared fixtures

From any machine (no GPU needed):

```bash
python PhoneCacheBlend/scripts/generate_quality_fixtures.py
```

Writes:

- `CacheBlend/inputs/quality_simple.json`
- `CacheBlend/inputs/quality_harder.json`

These mirror `QualityTestCases.swift` (33 passages, 10 simple queries, 3 harder queries).

## Step 2 — Run original CacheBlend (CUDA machine)

```bash
cd CacheBlend
# One-time setup (see CacheBlend/README.md):
#   cd vllm_blend && pip install -e . && cd .. && pip install -r requirements.txt

python example/blend_quality_suite.py --suite simple
python example/blend_quality_suite.py --suite harder
```

Outputs:

- `CacheBlend/results/cacheblend_simple.json`
- `CacheBlend/results/cacheblend_harder.json`

Options:

- `--recomp-ratio 0.18` (default, matches PhoneCacheBlend)
- `--max-queries 1` for a quick smoke test
- `--model mistralai/Mistral-7B-Instruct-v0.2`

## Step 3 — Run PhoneCacheBlend (iOS app)

1. Load Qwen2.5-1.5B in the app.
2. Open **Quality Test** → pick **Simple_test** or **Harder_test**.
3. Run **Run All (baseline + PCB)**.
4. Copy scores and TTFT from the log into `PhoneCacheBlend/results/phone_simple.json` (use `phone_results_template.json` as a guide), or record manually.

## Step 4 — Compare results

```bash
python PhoneCacheBlend/scripts/compare_quality_results.py \
  --reference CacheBlend/results/cacheblend_simple.json \
  --phone PhoneCacheBlend/results/phone_simple.json
```

The table shows per-query baseline vs fuse scores and TTFT for both platforms.

## Also run upstream WikiMQA benchmark (optional)

The paper’s default eval is separate from our custom fixtures:

```bash
cd CacheBlend
python example/blend_wikimqa.py
```

That uses 2WikiMQA passages, Mistral-7B, and **token F1** (not key-phrase scoring). Use it to sanity-check your CacheBlend install, not for direct PhoneCacheBlend comparison.

## Environment note (this Mac)

CacheBlend requires **CUDA + PyTorch**. This M2 Mac cannot run `blend_quality_suite.py` locally. Use a cloud GPU (A40/A100) or a Linux workstation with NVIDIA drivers.

## Canonical test data

- Swift app: `app/llama.swiftui/Models/QualityTestCases.swift`
- Python export: `scripts/quality_suite_data.py`

If you change passages or queries, update `quality_suite_data.py` and re-run `generate_quality_fixtures.py`.
