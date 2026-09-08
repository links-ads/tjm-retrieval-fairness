import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


def _resample_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[indices].mean(axis=1)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean of per-query metric values.

    Args:
        values: One metric value per query.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the PCG64 generator, so intervals are reproducible.
        level: Confidence level.

    Returns:
        The observed mean, the lower bound, and the upper bound.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("bootstrap_ci requires at least one value")

    means = _resample_means(array, n_resamples, seed)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(array.mean()), float(lo), float(hi)


def bootstrap_std(
    values: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Standard deviation of the bootstrap distribution of the mean.

    This is a compact alternative to ``bootstrap_ci``'s interval for the same quantity: how much our
    estimate of the mean could plausibly shift given the finite query sample. It answers a different
    question from the per-query standard deviation of the raw values, which describes how much
    performance varies from one query to the next and does not shrink as the query count grows.

    Args:
        values: One metric value per query.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the PCG64 generator, so results are reproducible.

    Returns:
        The standard deviation of the resampled means.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("bootstrap_std requires at least one value")

    return float(_resample_means(array, n_resamples, seed).std())


def paired_bootstrap_test(
    a: list[float],
    b: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap test on per-query metric values for two systems.

    Query indices are resampled once per iteration and applied to both systems, so the pairing that
    makes the comparison sensitive is preserved. The p-value comes from the centred bootstrap
    distribution, counting resamples whose deviation from the observed delta is at least as large as
    the observed delta itself.

    Args:
        a: Per-query values for the first system.
        b: Per-query values for the second system, aligned query-by-query with ``a``.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the PCG64 generator.
        level: Confidence level for the interval on the delta.

    Returns:
        Maps "delta", "ci_low", "ci_high", "p_value" and "n_queries" to their values.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"paired systems must cover the same queries, got {first.shape} and {second.shape}")
    if first.size == 0:
        raise ValueError("paired_bootstrap_test requires at least one value")

    differences = first - second
    delta = float(differences.mean())

    resampled = _resample_means(differences, n_resamples, seed)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(resampled, [tail, 1.0 - tail])
    p_value = float(np.mean(np.abs(resampled - delta) >= abs(delta)))

    return {
        "delta": delta,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_value,
        "n_queries": float(first.size),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down correction for a family of hypotheses.

    Args:
        p_values: Uncorrected p-values, in the caller's own order.
        alpha: Family-wise error rate.

    Returns:
        corrected: Adjusted p-values in the input order.
        reject: Whether each hypothesis is rejected at ``alpha``, in the input order.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    running = 0.0

    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p_values[index]))
        corrected[index] = running

    return corrected, [corrected[i] <= alpha for i in range(n)]
