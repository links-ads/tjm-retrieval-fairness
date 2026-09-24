import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_array, hstack, vstack

from .significance import DEFAULT_SEED

SUPPORT_TOLERANCE = 1e-9


def _position_discounts(size: int) -> np.ndarray:
    return 1.0 / np.log2(1.0 + np.arange(1, size + 1))


def _normalised_merit(window: list[str], scores: dict[str, float]) -> np.ndarray:
    raw = np.array([scores[candidate] for candidate in window], dtype=float)
    spread = raw.max() - raw.min()
    return np.ones_like(raw) if spread <= 0.0 else (raw - raw.min()) / spread


def _solve_exposure_lp(merit: np.ndarray, is_group_a: np.ndarray) -> np.ndarray:
    """Utility-maximal doubly stochastic placement at equal exposure per unit of merit.

    Solved lexicographically, for the smallest achievable violation of the parity constraint and then
    for the best utility attaining it, since exposure is capped by the first position's discount while
    a merit ratio is not.
    """
    size = merit.size
    total_a, total_b = merit[is_group_a].sum(), merit[~is_group_a].sum()
    if total_a <= 0.0 or total_b <= 0.0:
        raise ValueError("both groups need positive total merit for the exposure ratio to be defined")

    discounts = _position_discounts(size)
    parity = np.outer(np.where(is_group_a, 1.0 / total_a, -1.0 / total_b), discounts).ravel()
    cells, ones = np.arange(size * size), np.ones(size * size)
    shared = {
        "A_eq": hstack([
            vstack([coo_array((ones, (cells // size, cells)), shape=(size, size * size)),
                    coo_array((ones, (cells % size, cells)), shape=(size, size * size))]),
            coo_array((2 * size, 1)),
        ]),
        "b_eq": np.ones(2 * size),
        "A_ub": coo_array(np.vstack([np.append(parity, -1.0), np.append(-parity, -1.0)])),
        "b_ub": np.zeros(2),
        "method": "highs",
    }
    bounds = [(0.0, 1.0)] * (size * size)

    nearest = linprog(c=np.append(np.zeros(size * size), 1.0), bounds=bounds + [(0.0, None)], **shared)
    if not nearest.success:
        raise ValueError(f"exposure LP did not solve: {nearest.message}")

    best = linprog(c=np.append(-np.outer(merit, discounts).ravel(), 0.0),
                   bounds=bounds + [(0.0, float(nearest.x[-1]) + SUPPORT_TOLERANCE)], **shared)
    if not best.success:
        raise ValueError(f"exposure LP did not solve: {best.message}")
    return best.x[:-1].reshape(size, size)


def _birkhoff_decompose(matrix: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Weighted permutations summing to the given doubly stochastic matrix, by repeated matching."""
    residual = np.array(matrix, dtype=float)
    size = residual.shape[0]
    terms: list[tuple[float, np.ndarray]] = []

    for _ in range(size * size - size + 2):
        if residual.sum() <= SUPPORT_TOLERANCE * size:
            break
        rows, columns = linear_sum_assignment(residual, maximize=True)
        weight = float(residual[rows, columns].min())
        if weight <= SUPPORT_TOLERANCE:
            break
        residual[rows, columns] -= weight
        terms.append((weight, columns))

    total = sum(weight for weight, _ in terms)
    return [(weight / total, permutation) for weight, permutation in terms] if total else terms


def exposure_lp_rerank(
    ranked: list[str],
    scores: dict[str, float],
    group_of: dict[str, str],
    group_a: str,
    group_b: str,
    k: int,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Re-rank the top-``k`` so the two groups receive equal exposure per unit of estimated merit.

    The merit-aware counterpart to :func:`~tjm.metrics.fairness_ranking.fair_rerank`: exposure is
    allocated in proportion to the model's own scores rather than to group counts, by solving the
    Singh and Joachims exposure program and sampling a ranking from a Birkhoff--von Neumann
    decomposition of the solution. Merit comes from the scores rather than the relevance labels, so
    the intervention uses only what the ranker knows at query time, exactly as FA*IR does.

    Unlabelled candidates are dropped, matching ``fair_rerank``, so the two interventions are scored
    over the same candidate set. A window holding only one group, or one whose groups cannot both
    carry positive merit, has no exposure ratio to equalise and is returned in score order.

    Args:
        ranked: Candidate ids, best first.
        scores: Maps candidate id to the model score behind that ranking.
        group_of: Maps candidate id to group label; candidates may be absent.
        group_a: First group in the exposure comparison.
        group_b: Second group in the exposure comparison.
        k: Length of the window that is re-ranked.
        seed: Seed for sampling a permutation from the decomposition.

    Returns:
        The re-ranked window, holding only labelled candidates.
    """
    window = [candidate for candidate in ranked if group_of.get(candidate) in {group_a, group_b}][:k]
    if not window:
        return []

    merit = _normalised_merit(window, scores)
    is_group_a = np.array([group_of[candidate] == group_a for candidate in window])
    if merit[is_group_a].sum() <= 0.0 or merit[~is_group_a].sum() <= 0.0:
        return window

    terms = _birkhoff_decompose(_solve_exposure_lp(merit, is_group_a))
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = terms[rng.choice(len(terms), p=[weight for weight, _ in terms])][1]
    return [window[i] for i in np.argsort(chosen)]
