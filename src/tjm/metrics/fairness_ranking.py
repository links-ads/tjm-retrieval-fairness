import math
from collections import Counter

from scipy.stats import binom

# A group absent from a prefix would give an infinite log-ratio, which no aggregate survives; the
# floor is the skew of one item in a 10,000-long list, well below any value the data produces.
SKEW_FLOOR = math.log(1e-4)


def _pool_distribution(ranked: list[str], group_of: dict[str, str]) -> dict[str, float]:
    labelled = [group_of[item] for item in ranked if item in group_of]
    counts = Counter(labelled)
    return {group: counts[group] / len(labelled) for group in counts} if labelled else {}


def skew_at_k(
    ranked: list[str],
    group_of: dict[str, str],
    k: int,
    desired: dict[str, float] | None = None,
) -> dict[str, float]:
    """Log-ratio of each group's share of the top-``k`` to its desired share.

    Zero means the shortlist reflects the target distribution, positive means the group is
    over-represented in it and negative under-represented. Following Geyik et al., the target defaults
    to the group's proportion in the ranked pool itself, so the measure isolates what the ranking does
    rather than restating the population imbalance.

    Args:
        ranked: Candidate ids, best first.
        group_of: Maps candidate id to group label; unlabelled candidates are ignored.
        k: Cutoff defining the shortlist.
        desired: Target proportion per group. Defaults to the pool proportions.

    Returns:
        Maps each group present in the target distribution to its skew.
    """
    target = desired if desired is not None else _pool_distribution(ranked, group_of)
    shortlist = [group_of[item] for item in ranked[:k] if item in group_of]
    counts = Counter(shortlist)
    total = len(shortlist)
    skews = {}
    for group, share in target.items():
        if share <= 0:
            continue
        observed = counts[group] / total if total else 0.0
        skews[group] = max(math.log(observed / share), SKEW_FLOOR) if observed > 0 else SKEW_FLOOR
    return skews


def ndkl(
    ranked: list[str],
    group_of: dict[str, str],
    desired: dict[str, float] | None = None,
) -> float:
    """Normalised discounted cumulative KL-divergence of the prefix distributions from the target.

    Where skew inspects one cutoff, this scores every prefix and discounts by position, so a group
    pushed below the fold is penalised even when the top-``k`` looks balanced. It is non-negative and
    zero only when every prefix matches the target distribution.

    Args:
        ranked: Candidate ids, best first.
        group_of: Maps candidate id to group label; unlabelled candidates are ignored.
        desired: Target proportion per group. Defaults to the pool proportions.

    Returns:
        The NDKL of the ranking; lower is closer to the target at every depth.
    """
    labelled = [group_of[item] for item in ranked if item in group_of]
    if not labelled:
        return 0.0
    target = desired if desired is not None else _pool_distribution(ranked, group_of)

    counts = Counter()
    total_divergence = 0.0
    normaliser = 0.0
    for position, group in enumerate(labelled, start=1):
        counts[group] += 1
        divergence = 0.0
        for label, share in target.items():
            if share <= 0:
                continue
            observed = counts[label] / position
            if observed > 0:
                divergence += observed * math.log(observed / share)
        weight = 1.0 / math.log2(position + 1)
        total_divergence += weight * divergence
        normaliser += weight
    return total_divergence / normaliser if normaliser else 0.0


def _exposure(rank: int) -> float:
    return 1.0 / math.log2(1.0 + rank)


def exposure_ratios(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    group_of: dict[str, str],
    group_a: str,
    group_b: str,
    group_side: str = "candidate",
) -> dict[str, float | None]:
    """Disparate treatment and disparate impact ratios between two groups.

    Both are merit-aware in the sense of Singh and Joachims: exposure is compared \\emph{per unit of
    relevance}, so a group that is genuinely less relevant to the queries is not expected to receive
    equal exposure. A ratio of one means exposure (or click-through) is allocated in proportion to
    relevance for both groups; above one favours ``group_a``.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids.
        group_of: Maps candidate id to group label.
        group_a: Group placed in the numerator.
        group_b: Group placed in the denominator.
        group_side: Which side carries the attribute, ``"candidate"`` or ``"query"``. With
            ``"query"`` every (query, candidate) pair inherits the query's group; note that DTR is
            then close to vacuous whenever all queries share a pool size, because the exposure summed
            over a fixed-length ranking is a constant and the ratio reduces to the inverse ratio of
            labelled relevance.

    Returns:
        Maps "dtr" and "dir" to the two ratios, ``None`` when a group has no relevant members, plus
        the per-group exposure, relevance and click-through means behind them.
    """
    if group_side not in {"candidate", "query"}:
        raise ValueError(f"group_side must be 'candidate' or 'query', got {group_side!r}")
    totals = {
        group: {"exposure": 0.0, "relevance": 0.0, "click": 0.0, "n": 0}
        for group in (group_a, group_b)
    }
    for query_id, ranked in rankings.items():
        relevant = set(references.get(query_id, []))
        for rank, candidate_id in enumerate(ranked, start=1):
            group = group_of.get(query_id if group_side == "query" else candidate_id)
            if group not in totals:
                continue
            value = _exposure(rank)
            label = 1.0 if candidate_id in relevant else 0.0
            bucket = totals[group]
            bucket["exposure"] += value
            bucket["relevance"] += label
            bucket["click"] += value * label
            bucket["n"] += 1

    means = {}
    for group, bucket in totals.items():
        n = bucket["n"] or 1
        means[group] = {key: bucket[key] / n for key in ("exposure", "relevance", "click")}

    def ratio(numerator: str) -> float | None:
        first, second = means[group_a], means[group_b]
        if first["relevance"] <= 0 or second["relevance"] <= 0:
            return None
        denominator = second[numerator] / second["relevance"]
        if denominator <= 0:
            return None
        return (first[numerator] / first["relevance"]) / denominator

    result: dict[str, float | None] = {"dtr": ratio("exposure"), "dir": ratio("click")}
    for group in (group_a, group_b):
        for key in ("exposure", "relevance", "click"):
            result[f"{key}_{group}"] = means[group][key]
    return result


def amortized_attention(
    rankings: dict[str, list[str]],
    references: dict[str, list[str]],
    group_of: dict[str, str],
) -> dict[str, float]:
    """Equity of attention accumulated across the whole query set.

    A single ranking must put someone first, so fairness is only achievable in aggregate: over many
    queries each candidate's share of cumulative attention should match its share of cumulative
    relevance. The unfairness is the L1 distance between those two distributions, in the sense of
    Biega et al., and is reported alongside each group's two shares.

    Args:
        rankings: Maps query id to its ranked candidate ids, best first.
        references: Maps query id to its relevant candidate ids.
        group_of: Maps candidate id to group label.

    Returns:
        Maps "unfairness" to the L1 distance, and ``attention_share_<group>`` and
        ``relevance_share_<group>`` to each group's share of the two totals.
    """
    attention: dict[str, float] = {}
    relevance: dict[str, float] = {}
    for query_id, ranked in rankings.items():
        relevant = set(references.get(query_id, []))
        for rank, candidate_id in enumerate(ranked, start=1):
            if candidate_id not in group_of:
                continue
            attention[candidate_id] = attention.get(candidate_id, 0.0) + _exposure(rank)
            relevance[candidate_id] = relevance.get(candidate_id, 0.0) + (
                1.0 if candidate_id in relevant else 0.0
            )

    attention_total = sum(attention.values())
    relevance_total = sum(relevance.values())
    if attention_total <= 0 or relevance_total <= 0:
        return {"unfairness": 0.0}

    unfairness = sum(
        abs(attention.get(candidate_id, 0.0) / attention_total - relevance.get(candidate_id, 0.0) / relevance_total)
        for candidate_id in attention
    )
    result = {"unfairness": unfairness}
    for group in sorted(set(group_of.values())):
        members = [candidate_id for candidate_id in attention if group_of[candidate_id] == group]
        result[f"attention_share_{group}"] = sum(attention[c] for c in members) / attention_total
        result[f"relevance_share_{group}"] = sum(relevance[c] for c in members) / relevance_total
    return result


def fair_rerank(
    ranked: list[str],
    group_of: dict[str, str],
    protected: str,
    p: float,
    k: int,
    alpha: float = 0.1,
) -> list[str]:
    """Re-rank the top-``k`` so the protected group meets a minimum proportion at every prefix.

    This is the FA*IR construction of Zehlike et al.: the required number of protected candidates at
    each prefix is the ``alpha`` quantile of a binomial with success probability ``p``, and the
    ranking is filled greedily from two score-ordered queues, taking a protected candidate whenever
    the prefix would otherwise fall below that floor. Within-group order is never changed, so utility
    is given up only where the constraint binds. The significance level is used unadjusted, without
    the paper's correction for testing every prefix.

    Args:
        ranked: Candidate ids, best first.
        group_of: Maps candidate id to group label.
        protected: The group receiving the floor.
        p: Target minimum proportion for that group.
        k: Length of the re-ranked list.
        alpha: Significance level for the binomial floor; larger demands more protected candidates.

    Returns:
        The re-ranked top-``k``, or as many candidates as exist.
    """
    protected_queue = [c for c in ranked if group_of.get(c) == protected]
    other_queue = [c for c in ranked if c in group_of and group_of[c] != protected]
    target = min(k, len(protected_queue) + len(other_queue))
    minimum = [int(binom.ppf(alpha, position, p)) for position in range(target + 1)]

    result: list[str] = []
    taken_protected = 0
    protected_index = other_index = 0
    for position in range(1, target + 1):
        needs_protected = taken_protected < minimum[position]
        if needs_protected and protected_index < len(protected_queue):
            result.append(protected_queue[protected_index])
            protected_index += 1
            taken_protected += 1
            continue
        if other_index >= len(other_queue):
            result.append(protected_queue[protected_index])
            protected_index += 1
            taken_protected += 1
            continue
        if protected_index < len(protected_queue) and ranked.index(
            protected_queue[protected_index]
        ) < ranked.index(other_queue[other_index]):
            result.append(protected_queue[protected_index])
            protected_index += 1
            taken_protected += 1
        else:
            result.append(other_queue[other_index])
            other_index += 1
    return result
