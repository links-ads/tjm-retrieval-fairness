"""Offline analysis driver for the ECIR resubmission.

Rescoring every saved run costs no GPU time: predictions are exhaustive over the full candidate
pool, so hit@k / recall@k / nDCG@k / MRR, bootstrap confidence intervals, Holm-corrected significance
against BM25, and retrieve-then-rerank cascades are all reconstructed from disk.

    uv run tools/analyze.py --runs-dir outputs/zeroshot --split triplets --data-dir data

Outputs land under ``--output-dir`` (default outputs/analysis):
    per_query_metrics.csv   every run x every query x every metric
    summary.csv             means with 95% bootstrap CIs, one row per (system, mode, pool)
    significance.csv        Holm-corrected contrasts: every system vs BM25, every cascade vs its
                             own bare first stage
    cascade_paper.csv       the 24 (first_stage, reranker, depth) combinations reported in the paper
    cascade_appendix.csv    the remaining combinations, reported for completeness
    tables/*.tex            LaTeX for the tables above
"""

import argparse
import csv
from pathlib import Path

from tjm.metrics import (
    align_per_query,
    build_cascade_grid,
    build_references,
    compare_against_baseline,
    correct_family,
    discover_runs,
    summarise_runs,
)
from tjm.metrics.significance import DEFAULT_RESAMPLES, DEFAULT_SEED, paired_bootstrap_test

EVALUATION_MODES = ("job_to_talent", "talent_to_job")
CANDIDATE_POOLS = ("full", "observed")
CASCADE_DEPTHS = (10, 20, 50, 100)
CONTRAST_METRICS = ("hit@1", "recall@1", "recall@5", "ndcg@5", "hit@10", "recall@10", "recall@20", "ndcg@10", "mrr")
BASELINE_SYSTEM = "bm25"

# The paper's benchmarked models (Table 4/5); everything else discovered on disk goes to the
# appendix cascade table, since it costs nothing extra to report.
PAPER_FIRST_STAGES = ("bm25", "mpnet", "minilm", "sheared_llama", "snowflake", "qwen3_embedding")
PAPER_RERANKERS = ("jina", "qwen", "qwen4", "qwen8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    for row in rows[1:]:
        columns += [key for key in row if key not in columns]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_latex_table(rows: list[dict], columns: list[str], caption: str, label: str) -> str:
    header = " & ".join(columns) + r" \\"
    lines = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{'l' * len(columns)}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )


def run_analysis(
    runs_dir: Path,
    split: str,
    data_dir: str,
    output_dir: Path,
    n_resamples: int,
    seed: int,
    k_values: list[int],
) -> None:
    per_query_csv_rows: list[dict] = []
    summary_rows: list[dict] = []
    significance_rows: list[dict] = []
    per_system_per_query: dict[tuple, dict] = {}

    for mode in EVALUATION_MODES:
        references = build_references(data_dir, split, mode)

        for pool in CANDIDATE_POOLS:
            runs = discover_runs(runs_dir, split, evaluation_mode=mode, candidate_pool=pool)
            if not runs:
                continue

            per_query_out: dict = {}
            rows = summarise_runs(
                runs, references, k_values, n_resamples=n_resamples, seed=seed, per_query_out=per_query_out
            )
            summary_rows.extend(rows)

            for (system, run_mode, run_pool), per_query in per_query_out.items():
                per_system_per_query[(system, run_mode, run_pool)] = per_query
                for query_id, values in per_query.items():
                    per_query_csv_rows.append(
                        {
                            "system": system,
                            "evaluation_mode": run_mode,
                            "candidate_pool": run_pool,
                            "query_id": query_id,
                            **values,
                        }
                    )

            if BASELINE_SYSTEM in {key[0] for key in per_query_out}:
                systems_by_metric = {name: pq for (name, _, _), pq in per_query_out.items()}
                for metric in CONTRAST_METRICS:
                    try:
                        contrasts = compare_against_baseline(
                            systems_by_metric,
                            baseline=BASELINE_SYSTEM,
                            metric=metric,
                            n_resamples=n_resamples,
                            seed=seed,
                        )
                    except ValueError:
                        continue
                    for contrast in contrasts:
                        row = {"contrast_type": "vs_baseline", "evaluation_mode": mode, "candidate_pool": pool}
                        significance_rows.append({**row, **contrast})

    # Cascade grid: reconstructed only over the full pool, since a shortlist-then-rerank cascade is
    # only meaningful when the first stage is scored over the whole candidate catalogue. Every
    # discovered system can act as either role, since discover_runs does not distinguish retrievers
    # from rerankers.
    all_systems = sorted({key[0] for key in discover_runs(runs_dir, split, candidate_pool="full")})

    cascade_paper_rows: list[dict] = []
    cascade_appendix_rows: list[dict] = []
    # Raw (uncorrected) cascade-vs-first-stage contrasts, grouped by the family Holm correction must
    # run over: every cascade sharing an (evaluation_mode, metric) is one research question, even
    # though each has its own baseline (its own bare first stage), so no single call to
    # compare_against_baseline sees the whole family. Collect first, correct once per family below.
    raw_cascade_contrasts: dict[tuple[str, str], list[dict]] = {}

    for mode in EVALUATION_MODES:
        references = build_references(data_dir, split, mode)
        cascade_per_query: dict = {}
        grid = build_cascade_grid(
            runs_dir,
            split=split,
            first_stage_systems=all_systems,
            reranker_systems=all_systems,
            depths=list(CASCADE_DEPTHS),
            evaluation_mode=mode,
            references=references,
            k_values=k_values,
            n_resamples=n_resamples,
            seed=seed,
            per_query_out=cascade_per_query,
        )

        for row in grid:
            row["evaluation_mode"] = mode
            is_paper = row["first_stage"] in PAPER_FIRST_STAGES and row["reranker"] in PAPER_RERANKERS
            (cascade_paper_rows if is_paper else cascade_appendix_rows).append(row)

            # A reranker cascading over its own output is not a cascade; skip the degenerate pair
            # rather than let it dilute the correction with a trivial delta=0 hypothesis.
            first_stage, reranker, depth = row["first_stage"], row["reranker"], row["depth"]
            if first_stage == reranker:
                continue

            bare = per_system_per_query.get((first_stage, mode, "full"))
            cascade_values = cascade_per_query.get((first_stage, reranker, depth))
            if bare is None or cascade_values is None:
                continue
            for metric in CONTRAST_METRICS:
                try:
                    query_ids, values = align_per_query({"bare": bare, "cascade": cascade_values}, metric)
                except ValueError:
                    continue
                if len(query_ids) < 2:
                    continue
                test = paired_bootstrap_test(values["cascade"], values["bare"], n_resamples=n_resamples, seed=seed)
                raw_cascade_contrasts.setdefault((mode, metric), []).append(
                    {
                        "contrast_type": "cascade_vs_first_stage",
                        "evaluation_mode": mode,
                        "candidate_pool": "full",
                        "metric": metric,
                        "system": f"{first_stage}->{reranker}",
                        "baseline": first_stage,
                        "first_stage": first_stage,
                        "reranker": reranker,
                        "depth": depth,
                        "n_queries": len(query_ids),
                        "delta": test["delta"],
                        "ci_low": test["ci_low"],
                        "ci_high": test["ci_high"],
                        "p_value": test["p_value"],
                    }
                )

    for family_rows in raw_cascade_contrasts.values():
        significance_rows.extend(correct_family(family_rows))

    write_csv(output_dir / "per_query_metrics.csv", per_query_csv_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "significance.csv", significance_rows)
    write_csv(output_dir / "cascade_paper.csv", cascade_paper_rows)
    write_csv(output_dir / "cascade_appendix.csv", cascade_appendix_rows)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        columns = [
            "system", "evaluation_mode", "candidate_pool", "n_queries",
            "hit@10", "recall@10", "recall@20", "ndcg@10", "mrr",
        ]
        caption = "Retrieval results with Hit@k, Recall@k, nDCG@10 and MRR."
        (tables_dir / "summary.tex").write_text(format_latex_table(summary_rows, columns, caption, "tab:summary"))
    if cascade_paper_rows:
        columns = [
            "first_stage", "reranker", "depth", "evaluation_mode",
            "first_stage_recall", "hit@10", "recall@10", "ndcg@10", "mrr",
        ]
        caption = "Retrieve-then-rerank cascades."
        cascade_tex = format_latex_table(cascade_paper_rows, columns, caption, "tab:cascade")
        (tables_dir / "cascade.tex").write_text(cascade_tex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=str, default="outputs/zeroshot")
    parser.add_argument("--split", type=str, default="triplets")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis")
    parser.add_argument("--k", type=str, default="1,5,10,20,50,100")
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    k_values = [int(x) for x in args.k.split(",") if x]
    run_analysis(
        runs_dir=Path(args.runs_dir),
        split=args.split,
        data_dir=args.data_dir,
        output_dir=Path(args.output_dir),
        n_resamples=args.n_resamples,
        seed=args.seed,
        k_values=k_values,
    )
    print(f"Analysis written to {args.output_dir}")
