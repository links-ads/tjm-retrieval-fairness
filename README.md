# Dense Retrieval for Talent-Job Matching

Code and experimental output behind the results section of the paper: zero-shot retrieval and
reranking, retrieve-then-rerank cascades, and ranking fairness on the InnoNext talent-job matching
(TJM) dataset.

Every number in `paper/results.tex` is reproducible from this repository without a GPU. The saved
rankings are exhaustive over the candidate pool, so metrics, bootstrap intervals, significance tests,
cascades and fairness measures are all reconstructed offline from disk.

> **This repository contains personal data.** `data/resumes.csv`, `data/triplets.csv` and
> `data/talents_demographics.csv` hold resume text, application and hiring outcomes, and
> gender/age/nationality for real job applicants. Identifiers are pseudonymous, which does not make
> the data non-personal. Keep this repository private, and do not redistribute the contents of
> `data/` without the data controller's approval.

## Quickstart

```bash
uv sync
uv run pytest tests/ -q
```

Reproduce the three analyses (about 15 minutes total, CPU only):

```bash
# zero-shot metrics, bootstrap CIs, Holm-corrected significance vs BM25, and the cascade grid
uv run tools/analyze.py --runs-dir outputs/zeroshot --split triplets --data-dir data

# group fairness, both directions and both pools
uv run tools/analyze_fairness.py --systems bm25 mpnet sheared_llama qwen3_embedding qwen8

# ranking fairness (Skew, NDKL, DTR, DIR, amortized attention, FA*IR), Job->Talent
uv run tools/analyze_ranking_fairness.py
```

Each writes into `outputs/`, overwriting the committed results with identical values. To rebuild
the paper's fairness table and check it against `paper/results.tex`:

```bash
uv run python paper/scripts/make_fairness_ranking_table.py
```

## Layout

```
src/tjm/metrics/     the analysis layer
  ranking.py           Recall@k, nDCG@k, Hit@k, MRR, per-query and aggregate
  report.py            run discovery, relevance judgements, Holm-corrected contrasts vs a baseline
  significance.py      bootstrap CIs, bootstrap std of the mean, paired bootstrap tests
  cascade.py           exact reconstruction of a retrieve-then-rerank cascade from two saved runs
  cascade_grid.py      the (first stage x reranker x depth) grid
  fairness.py          group fairness: outcome rates by group, permutation tests, representation gaps
  fairness_ranking.py  Skew@k, NDKL, DTR/DIR exposure ratios, amortized attention, FA*IR reranking
src/tjm/models/      BM25 and the retrievers/rerankers, for regenerating rankings
src/tjm/data/        dataset loading and relevance construction
tools/               the three analysis entrypoints
tests/               tests over the analysis layer
data/                dataset CSVs (see the warning above)
outputs/zeroshot/    saved rankings, gzipped, per system / split / direction / pool
outputs/analysis/    zero-shot and cascade results
outputs/fairness_analysis/  fairness results
paper/results.tex    the results section
paper/scripts/       regenerate the paper's tables and figures
  make_fairness_ranking_table.py   emits tab:fairness-ranking, verified row-for-row
  make_unified_table_std.py        emits tab:unified, verified row-for-row
  make_cascade_table.py            emits tab:cascade, verified row-for-row
  make_fairness_ranking_plot.py    the fairness forest plot
```

## Saved rankings

`outputs/zeroshot/<system>/triplets/<direction>_<pool>.json.gz` maps each query id to every
candidate scored, as `{"candidate_id": ..., "score": ...}`. They are stored gzipped: 283 MB of
pretty-printed JSON compresses to 26 MB. `load_run`, `discover_runs` and the cascade grid read
`.json` and `.json.gz` interchangeably, preferring a plain `.json` when both are present, so an
uncompressed working copy shadows the committed archive rather than being ignored.

Fifteen systems are included: `bm25`, `mpnet`, `minilm`, `sheared_llama`, `snowflake`, `bge`,
`bge-base`, `bge-large`, `qwen3_embedding`, `qwen3_embedding4`, `qwen3_embedding8` (retrievers) and
`jina`, `qwen`, `qwen4`, `qwen8` (rerankers). Rerankers score the full pool directly; the cascaded
setting is reconstructed from these same files.

All fifteen ship even though the paper's tables report a subset, because the Holm correction in
`significance.csv` runs over the whole family of systems tested against BM25. Dropping the systems
the paper does not tabulate would shrink that family and silently make the correction less
conservative, so the repository would no longer reproduce the paper's significance markers.

Two evaluation settings appear throughout:

- **Full-Pool** ranks every candidate in the corpus (735 talents, 320 vacancies).
- **Application-Pool** ranks only the candidates who actually appeared against that query, a much
  smaller and already-filtered pool.

Regenerating the rankings from scratch needs a GPU and the full dataset:

```bash
uv run -m tjm.evaluate --model qwen8 --split triplets --evaluation-mode job_to_talent
```

## Scope

This repository covers the paper's zero-shot, cascade and fairness results. The profile-enrichment
experiments with publication evidence, the PU-learning training code, and the fine-tuned checkpoints
live in the parent repository and are deliberately not included here.
