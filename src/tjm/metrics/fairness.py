import math
from collections import Counter

import numpy as np

from .significance import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci


def summarise_groups(
    values_by_group: dict[str, list[float]],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Per-group mean of a metric with a bootstrap interval on that mean.

    Args:
        values_by_group: One list of observations per group.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the PCG64 generator.

    Returns:
        One row per non-empty group, holding the group label, its size, mean and interval.
    """
    rows = []
    for group, values in values_by_group.items():
        if len(values) == 0:
            continue
        mean, low, high = bootstrap_ci(list(values), n_resamples=n_resamples, seed=seed)
        rows.append({"group": group, "n": len(values), "mean": mean, "ci_low": low, "ci_high": high})
    return rows


def permutation_group_test(
    a: list[float],
    b: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """Unpaired comparison of two groups' means, by permuting group labels.

    Group comparisons cannot use the paired test: the two samples are different queries (or
    different candidates), so there is no query-by-query correspondence to exploit. The null here is
    that the group label carries no information, which is generated exactly by reshuffling the labels
    over the pooled observations. The interval on the gap comes from resampling within each group,
    which keeps the group sizes fixed.

    Args:
        a: Observations for the first group.
        b: Observations for the second group.
        n_resamples: Number of permutations, and of bootstrap resamples for the interval.
        seed: Seed for the PCG64 generator, so results are reproducible.
        level: Confidence level for the interval on the gap.

    Returns:
        Maps "gap" (mean of ``a`` minus mean of ``b``), "ci_low", "ci_high", "p_value", "n_a" and
        "n_b" to their values.

    Raises:
        ValueError: If either group is empty.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.size == 0 or second.size == 0:
        raise ValueError("permutation_group_test requires at least one value in each group")

    gap = float(first.mean() - second.mean())
    rng = np.random.Generator(np.random.PCG64(seed))

    pooled = np.concatenate([first, second])
    n_a = first.size
    shuffled = pooled[np.argsort(rng.random((n_resamples, pooled.size)), axis=1)]
    null_gaps = shuffled[:, :n_a].mean(axis=1) - shuffled[:, n_a:].mean(axis=1)
    p_value = float(np.mean(np.abs(null_gaps) >= abs(gap)))

    resampled = (
        first[rng.integers(0, n_a, size=(n_resamples, n_a))].mean(axis=1)
        - second[rng.integers(0, second.size, size=(n_resamples, second.size))].mean(axis=1)
    )
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])

    return {
        "gap": gap,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p_value,
        "n_a": int(n_a),
        "n_b": int(second.size),
    }


def relevant_candidate_outcomes(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    k: int,
) -> list[dict]:
    """One outcome row per (query, relevant candidate) pair.

    This is the unit of analysis for equal opportunity on the candidate side: among candidates that
    are actually relevant, does the ranker surface each group's as readily? Positives absent from the
    ranking are omitted, since a candidate that never entered the pool is not a ranking decision.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids.
        k: Cutoff defining retrieval.

    Returns:
        Rows holding the query id, candidate id, 1-based rank, whether the rank is within ``k``, and
        the reciprocal rank.
    """
    outcomes = []
    for query_id, ranked in rankings.items():
        positives = references.get(query_id)
        if not positives:
            continue
        position = {candidate_id: index + 1 for index, candidate_id in enumerate(ranked)}
        for candidate_id in positives:
            rank = position.get(candidate_id)
            if rank is None:
                continue
            outcomes.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "retrieved": 1.0 if rank <= k else 0.0,
                    "reciprocal_rank": 1.0 / rank,
                }
            )
    return outcomes


def top_k_group_shares(
    rankings: dict[str, list[str]],
    group_of: dict[str, str],
    k: int,
) -> dict[str, list[float]]:
    """Each group's share of the top-``k``, per query.

    Shares are taken over the labelled candidates in the top-``k`` rather than over ``k`` slots, so
    candidates with no recorded attribute do not deflate every group at once. Queries whose top-``k``
    contains no labelled candidate contribute nothing.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        group_of: Maps candidate id to its group label; candidates may be absent.
        k: Cutoff defining the shortlist.

    Returns:
        Maps every known group label to one share per contributing query.
    """
    universe = sorted(set(group_of.values()))
    shares: dict[str, list[float]] = {group: [] for group in universe}
    for ranked in rankings.values():
        labelled = [group_of[candidate_id] for candidate_id in ranked[:k] if candidate_id in group_of]
        if not labelled:
            continue
        counts = Counter(labelled)
        for group in universe:
            shares[group].append(counts[group] / len(labelled))
    return shares


def representation_gap(
    rankings: dict[str, list[str]],
    group_of: dict[str, str],
    k: int,
) -> dict[str, list[float]]:
    """Each group's top-``k`` share minus its share of the pool that top-``k`` was drawn from.

    Comparing one group's top-``k`` share against another's tests a fifty-fifty null, which in an
    unbalanced population merely restates the imbalance. The quantity of interest is instead whether
    the shortlist reflects the pool it came from, so this returns a per-query difference: zero means
    the top-``k`` mirrors that query's own pool, positive means the group is over-represented in it.
    Both sides are taken from the same query, so the resulting values are paired and a query is
    dropped from both sides together.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        group_of: Maps candidate id to its group label; candidates may be absent.
        k: Cutoff defining the shortlist.

    Returns:
        Maps every known group label to one top-``k``-minus-pool difference per contributing query.
    """
    universe = sorted(set(group_of.values()))
    gaps: dict[str, list[float]] = {group: [] for group in universe}
    for ranked in rankings.values():
        shortlist = [group_of[candidate_id] for candidate_id in ranked[:k] if candidate_id in group_of]
        pool = [group_of[candidate_id] for candidate_id in ranked if candidate_id in group_of]
        if not shortlist or not pool:
            continue
        shortlist_counts = Counter(shortlist)
        pool_counts = Counter(pool)
        for group in universe:
            gaps[group].append(shortlist_counts[group] / len(shortlist) - pool_counts[group] / len(pool))
    return gaps


def classification_rates(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    group_of: dict[str, str],
    k: int,
) -> dict[str, dict]:
    """Per-group rates for the group-fairness criteria, treating top-``k`` as a binary prediction.

    A (query, candidate) pair is the unit: the prediction is positive when the candidate is ranked
    within the top-``k`` for that query, and the label is positive when the candidate is relevant to
    it. The three criteria are then absolute differences between two groups' entries here: statistical
    parity over ``positive_rate``, equal opportunity over ``true_positive_rate``, and overall accuracy
    equality over ``accuracy``.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids; queries absent from it are skipped,
            since without judgements neither label nor accuracy is defined.
        group_of: Maps candidate id to its group label; candidates may be absent.
        k: Cutoff at which a candidate counts as predicted positive.

    Returns:
        Maps every known group label to its ``positive_rate``, ``true_positive_rate`` (``None`` when
        the group has no relevant members), ``accuracy``, ``n_candidates`` and ``n_relevant``.
    """
    return _rates(rankings, references, sorted(set(group_of.values())), k, lambda query_id, candidate_id: group_of.get(candidate_id))


def classification_rates_by_query_group(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    group_of: dict[str, str],
    k: int,
) -> dict[str, dict]:
    """Same rates as :func:`classification_rates`, but grouping pairs by the querying entity.

    Used where the protected attribute describes who is being served rather than who is being ranked:
    every (query, candidate) pair inherits the query's group, so the criteria compare how accurate the
    system's decisions are for one group of queriers against another.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids.
        group_of: Maps query id to its group label; queries may be absent, and are then skipped.
        k: Cutoff at which a candidate counts as predicted positive.

    Returns:
        As :func:`classification_rates`.
    """
    return _rates(rankings, references, sorted(set(group_of.values())), k, lambda query_id, candidate_id: group_of.get(query_id))


def _rates(rankings, references, universe, k, group_for):
    tally = {
        group: {"positives": 0, "true_positives": 0, "correct": 0, "n_candidates": 0, "n_relevant": 0}
        for group in universe
    }
    for query_id, ranked in rankings.items():
        positives = references.get(query_id)
        if not positives:
            continue
        relevant = set(positives)
        shortlist = set(ranked[:k])
        for candidate_id in ranked:
            group = group_for(query_id, candidate_id)
            if group is None:
                continue
            predicted = candidate_id in shortlist
            actual = candidate_id in relevant
            counters = tally[group]
            counters["n_candidates"] += 1
            counters["positives"] += int(predicted)
            counters["correct"] += int(predicted == actual)
            if actual:
                counters["n_relevant"] += 1
                counters["true_positives"] += int(predicted)

    rates = {}
    for group, counters in tally.items():
        total = counters["n_candidates"]
        relevant_total = counters["n_relevant"]
        rates[group] = {
            "positive_rate": counters["positives"] / total if total else None,
            "true_positive_rate": counters["true_positives"] / relevant_total if relevant_total else None,
            "accuracy": counters["correct"] / total if total else None,
            "n_candidates": total,
            "n_relevant": relevant_total,
        }
    return rates


def individual_classification_rates(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    group_of: dict[str, str],
    k: int,
    group_side: str = "candidate",
) -> dict[str, dict]:
    """Group rates built from per-person rates, so each protected individual counts once.

    Pooling every (query, candidate) pair weights a person by how many queries they appear in or are
    relevant to, which for a criterion about people is the wrong weighting: a talent relevant to nine
    vacancies would count nine times. Here each person's own rate is computed across the queries that
    concern them, and the group value is the mean over people. For the direction in which the person
    *is* the query this reduces to the ordinary macro-average over queries, which is how Recall@$k$ is
    reported elsewhere, so the two stay consistent.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids.
        group_of: Maps person id to group label. Ids are matched against candidates, and against the
            query itself when the query is the person.
        k: Cutoff at which a candidate counts as predicted positive.
        group_side: Which side of the ranking carries the attribute, ``"candidate"`` or ``"query"``.
            This is stated rather than inferred: query ids and candidate ids are drawn from one
            integer namespace here, so guessing per query lets an unlabelled query match a candidate
            id against the attribute table and invent an individual.

    Returns:
        Maps every known group label to its ``positive_rate``, ``true_positive_rate``, ``accuracy``,
        ``n_individuals`` and ``n_relevant_individuals``.

    Raises:
        ValueError: If ``group_side`` is neither ``"candidate"`` nor ``"query"``.
    """
    if group_side not in {"candidate", "query"}:
        raise ValueError(f"group_side must be 'candidate' or 'query', got {group_side!r}")
    values = individual_rate_values(rankings, references, group_of, k, group_side)
    mean = lambda observed: sum(observed) / len(observed) if observed else None
    return {
        group: {
            "positive_rate": mean(measures["positive_rate"]),
            "true_positive_rate": mean(measures["true_positive_rate"]),
            "accuracy": mean(measures["accuracy"]),
            "n_individuals": len(measures["positive_rate"]),
            "n_relevant_individuals": len(measures["true_positive_rate"]),
        }
        for group, measures in values.items()
    }


def individual_rate_values(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    group_of: dict[str, str],
    k: int,
    group_side: str = "candidate",
) -> dict[str, dict[str, list[float]]]:
    """Per-person rates, ungrouped, so group differences can be tested rather than only reported.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids.
        group_of: Maps person id to group label.
        k: Cutoff at which a candidate counts as predicted positive.
        group_side: ``"candidate"`` or ``"query"``, per :func:`individual_classification_rates`.

    Returns:
        Maps group label to ``positive_rate``, ``true_positive_rate`` and ``accuracy``, each a list
        holding one value per person in that group. The ``true_positive_rate`` list is shorter, since
        it covers only people who are relevant to at least one query.

    Raises:
        ValueError: If ``group_side`` is neither ``"candidate"`` nor ``"query"``.
    """
    if group_side not in {"candidate", "query"}:
        raise ValueError(f"group_side must be 'candidate' or 'query', got {group_side!r}")
    universe = sorted(set(group_of.values()))
    per_person: dict[str, dict[str, list[float]]] = {}
    for query_id, ranked in rankings.items():
        positives = references.get(query_id)
        if not positives:
            continue
        relevant = set(positives)
        shortlist = set(ranked[:k])
        for candidate_id in ranked:
            person = query_id if group_side == "query" else candidate_id
            if person not in group_of:
                continue
            predicted = candidate_id in shortlist
            actual = candidate_id in relevant
            bucket = per_person.setdefault(person, {"positive": [], "true_positive": [], "correct": []})
            bucket["positive"].append(float(predicted))
            bucket["correct"].append(float(predicted == actual))
            if actual:
                bucket["true_positive"].append(float(predicted))

    grouped: dict[str, dict[str, list[float]]] = {
        group: {"positive_rate": [], "true_positive_rate": [], "accuracy": []} for group in universe
    }
    for person, bucket in per_person.items():
        target = grouped[group_of[person]]
        target["positive_rate"].append(sum(bucket["positive"]) / len(bucket["positive"]))
        target["accuracy"].append(sum(bucket["correct"]) / len(bucket["correct"]))
        if bucket["true_positive"]:
            target["true_positive_rate"].append(sum(bucket["true_positive"]) / len(bucket["true_positive"]))
    return grouped


def median_split(
    values: dict[str, float],
    low_label: str,
    high_label: str,
    valid_range: tuple[float, float] | None = None,
) -> dict[str, str]:
    """Bucket a continuous attribute into two groups by its own median.

    A median split avoids an arbitrary cutoff for an attribute with no natural threshold. Values
    outside ``valid_range`` are dropped before the median is computed and are absent from the
    result, rather than being coerced into a group or allowed to shift the split point -- useful
    when the source column mixes real observations with sentinel or data-entry values (e.g. an age
    of 0).

    Args:
        values: Maps id to its raw value.
        low_label: Group label for values below the median.
        high_label: Group label for values at or above the median.
        valid_range: Inclusive (min, max); values outside it are excluded. Defaults to no filtering.

    Returns:
        Maps id to ``low_label`` or ``high_label``; ids outside the valid range are omitted.
    """
    kept = {
        id_: value
        for id_, value in values.items()
        if valid_range is None or valid_range[0] <= value <= valid_range[1]
    }
    if not kept:
        return {}
    ordered = sorted(kept.values())
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {id_: (high_label if value >= median else low_label) for id_, value in kept.items()}


def binary_category_split(values: dict[str, object], target: object, other_label: str) -> dict[str, str]:
    """Bucket a categorical attribute into the target category and everything else.

    Useful for a high-cardinality attribute (e.g. 74 distinct nationalities) with no natural binary
    split: comparing the single largest category against the pooled remainder uses all the
    non-missing data rather than discarding every id outside a hand-picked pair of categories.

    Args:
        values: Maps id to its raw category; missing values (``None`` or NaN) are dropped.
        target: The category placed in its own group.
        other_label: Label for every other non-missing category.

    Returns:
        Maps id to ``target`` (as a string) or ``other_label``; ids with a missing value are omitted.
    """
    def is_missing(value: object) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    return {
        id_: (str(target) if value == target else other_label)
        for id_, value in values.items()
        if not is_missing(value)
    }
