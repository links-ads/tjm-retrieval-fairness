# Fair Among Whom?

Code and saved rankings for *Fair Among Whom? Evaluating Gender Fairness in Job–Talent Matching
Across Candidate Scopes* — F. D'Asaro, M. Testa, J. J. Márquez Villacís, F. Dominici, G. Rizzo
(LINKS Foundation ADS; Politecnico di Torino DAUIN).

## Motivation

Talent–Job Matching (TJM) systems allocate access to employment, so fairness matters as much as
effectiveness. Yet TJM fairness is usually reported as a single number without stating the candidate
pool it was computed over. That omission is not cosmetic: ranking an entire talent catalogue and
ranking a vacancy's actual applicants are different problems, differing in group composition and in
relevance distribution, and the same measure over each can point in opposite directions. A fairness
number is uninterpretable without the pool it came from.

## Contributions

- **A dual-pool evaluation framework.** The candidate pool — FullPool (the whole catalogue) versus
  AppPool (only observed applications) — becomes an explicit evaluation axis: we formalise both,
  derive the query sets each induces, and state which fairness measures are admissible in each. The
  audit is restricted to Job→Talent, the only direction whose ranked items carry a protected
  attribute.
- **A fairness audit instantiating it.** Representation (Skew) and merit-aware (DTR, DIR) measures
  disagree across four architecturally distinct models; the disagreement reverses between pools
  because of who applies rather than any ranker; neither intervention resolves both views alone.
- **A real-world TJM dataset.** Recruitment outcomes from the InnoNext platform, with self-reported
  gender for 85.8% of ranked talents.

## Key result

In FullPool, Female talents are under-represented in the shortlist yet receive significantly *more*
exposure per unit of labelled relevance (DTR = 1.33–1.38). This reverses in AppPool, where DTR falls
to or below parity (0.89–0.94), identically across model types. The cause is the applicant funnel,
not the rankers: the Female share drops from 40.1% of all talents to 35.3% of those who apply to
30.3% within a job's applicant pool, and the Male-to-Female relevance ratio flips from 1.38 to 0.88.
FA*IR corrects Skew but shifts DTR inconsistently; the exposure LP moves DTR toward parity in
opposite directions per pool while degrading Skew. Neither resolves the trade-off.

## Data

Not distributed here. It holds resume text, hiring outcomes and self-reported demographics for real
applicants, so it is personal data under GDPR and is available only from the data controller. The
commands below expect it under `data/`.

## Usage

Saved rankings are exhaustive over each candidate pool, so every metric, interval, significance test,
cascade and fairness measure is reconstructed offline — no GPU needed.

```bash
uv sync

uv run tools/analyze.py --runs-dir outputs/zeroshot --split triplets --data-dir data
uv run tools/analyze_fairness.py --systems bm25 mpnet sheared_llama qwen3_embedding qwen8
uv run tools/analyze_ranking_fairness.py
uv run tools/analyze_exposure_feasibility.py
```

Each overwrites `outputs/` with identical values. Regenerating rankings from scratch needs a GPU:
`uv run -m tjm.evaluate --model qwen8 --split triplets --evaluation-mode job_to_talent`.

Fifteen systems ship though the paper tabulates a subset: the Holm correction runs over the whole
family tested against BM25, so dropping the rest would silently weaken it.

## Acknowledgment

Part of the InnoNext project, funded by the European Union's Horizon Europe research and innovation
programme under grant agreement No. 101160467. Views and opinions expressed are those of the authors
only and do not necessarily reflect those of the European Union.
