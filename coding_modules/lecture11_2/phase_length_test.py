"""
phase_length_test.py

Nonparametric permutation test comparing the observed distribution of
monotonic phase lengths in a time series to the expected distribution
under complete randomness (i.i.d. continuous; no trend, no persistence).

Summary
-------
1) Convert x_t to signs of first differences s_t = sign(x_{t+1} - x_t).
2) Monotonic phases are runs of identical signs; a run of k differences
   spans k+1 observations (phase length m >= 2).
3) Under i.i.d. continuous null, direction flips with probability q = 2/3.
   Therefore phase lengths (in observations) follow a shifted geometric:
       P(M = m) = (1 - q)^(m - 2) * q,  m >= 2,  with q = 2/3.
4) We compare the observed histogram to the theoretical PMF via a
   chi-square distance statistic and obtain a p-value by permutation:
   randomly permute the time order of x (preserves marginal distribution,
   removes serial dependence), recompute the statistic B times, and form
   a tail-area p-value.

Public API
----------
- phase_length_distribution_test(x, B=2000, q_null=2/3, K=None,
    drop_ties=True, seed=None) -> PhaseLengthTestResult
- monotonic_phase_lengths(x, drop_ties=True) -> List[int]

Example
-------
>>> from phase_length_test import phase_length_distribution_test
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> x = rng.normal(size=500)
>>> res = phase_length_distribution_test(x, B=2000, seed=1)
>>> res.p_perm
0.432  # (example)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict
import numpy as np

__all__ = [
    "PhaseLengthTestResult",
    "monotonic_phase_lengths",
    "phase_length_distribution_test",
]

# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class PhaseLengthTestResult:
    n: int                          # original length
    n_effective: int                # after dropping ties
    n_phases: int                   # number of monotonic phases
    bins: List[str]                 # labels: "2", "3", ..., "≥K"
    observed_counts: np.ndarray
    expected_counts: np.ndarray
    statistic: float                # chi-square distance to theoretical PMF
    p_perm: float                   # permutation p-value (>= tail)
    B: int                          # number of permutations
    q_null: float                   # 2/3 under iid continuous
    seed: Optional[int]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def monotonic_phase_lengths(
    x: Iterable[float],
    *,
    drop_ties: bool = True
) -> List[int]:
    """
    Compute monotonic phase lengths (in number of observations) of a time series.
    A 'phase' is a maximal strictly increasing or strictly decreasing run.

    Parameters
    ----------
    x : Iterable[float]
        Series values.
    drop_ties : bool, default True
        Compress consecutive equal values (classical iid-continuous assumption).

    Returns
    -------
    List[int]
        Phase lengths (each >= 2).
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size < 3:
        return []
    if drop_ties:
        mask = np.ones(arr.size, dtype=bool)
        mask[1:] = np.diff(arr) != 0
        arr = arr[mask]
        if arr.size < 3:
            return []

    d = np.diff(arr)
    d = d[d != 0]  # remove any residual zero differences
    if d.size == 0:
        return []

    s = np.sign(d)
    lengths_diff = []
    run = 1
    for i in range(1, s.size):
        if s[i] == s[i - 1]:
            run += 1
        else:
            lengths_diff.append(run)
            run = 1
    lengths_diff.append(run)

    # convert runs of differences to phase lengths in observations
    return [L + 1 for L in lengths_diff]


def _expected_pmf_shifted_geometric(K: int, *, q: float = 2/3) -> np.ndarray:
    """
    Theoretical PMF for phase lengths under iid continuous:
      P(m) = (1 - q)^(m - 2) * q for m = 2, 3, ...
    Return probabilities for bins: 2,3,...,K-1, '>=K' (tail).
    """
    if K < 3:
        raise ValueError("K must be >= 3 (minimal phase length is 2).")
    m_vals = np.arange(2, K)  # 2..K-1
    pmf = ((1 - q) ** (m_vals - 2)) * q
    tail = (1 - q) ** (K - 2)       # sum_{m>=K} (1 - q)^(m - 2) * q
    return np.append(pmf, tail)


def _auto_K(
    n_phases: int,
    max_obs_len: int,
    *,
    q: float = 2/3,
    min_tail_exp: float = 5.0
) -> int:
    """
    Choose K so the expected count in the '>=K' tail bin is at least `min_tail_exp`,
    but not exceeding the observed maximum length.
    """
    K = 2
    while n_phases * ((1 - q) ** (K - 2)) > min_tail_exp and K < max_obs_len:
        K += 1
    return max(3, min(K, max_obs_len))

# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

def phase_length_distribution_test(
    x: Iterable[float],
    *,
    B: int = 2000,
    q_null: float = 2/3,
    K: Optional[int] = None,
    drop_ties: bool = True,
    seed: Optional[int] = None,
) -> PhaseLengthTestResult:
    """
    Nonparametric permutation test of monotonic phase-length distribution vs. i.i.d.

    Test statistic:
        Chi-square distance between observed phase-length histogram and the
        theoretical shifted-geometric PMF implied by the i.i.d. continuous null:
            P(M = m) = (1 - q)^(m - 2) * q,  m >= 2,  q = 2/3.

    p-value:
        Obtained by permutation: randomly permute the time order of x (preserves the
        marginal distribution, removes serial dependence), recompute the statistic
        B times, and compute tail probability p = (1 + #{S_b >= S_obs}) / (B + 1).

    Parameters
    ----------
    x : Iterable[float]
        Time series (length >= 3 recommended).
    B : int, default 2000
        Number of permutations for the nonparametric p-value.
    q_null : float, default 2/3
        Flip probability under i.i.d. continuous; do not change unless justified.
    K : int, optional
        Upper cut for phase-length bins; bins are [2], [3], ..., [K-1], [>=K].
        If None, chosen automatically to keep expected tail count >= 5.
    drop_ties : bool, default True
        Compress consecutive equal values before computing phases.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PhaseLengthTestResult
        Counts, expected counts, chi-square statistic, and permutation p-value.
    """
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(list(x), dtype=float)
    n = x_arr.size
    if n < 3:
        raise ValueError("Input series must have length >= 3.")

    # Prepare a tie-compressed view length for reporting n_effective
    if drop_ties:
        mask = np.ones(n, dtype=bool)
        mask[1:] = np.diff(x_arr) != 0
        n_eff = int(mask.sum())
    else:
        n_eff = n

    # Observed phase lengths
    lengths = monotonic_phase_lengths(x_arr, drop_ties=drop_ties)
    if len(lengths) == 0:
        raise ValueError("No monotonic phases found (series too short or all equal after tie handling).")
    n_phases = len(lengths)
    max_len_obs = max(lengths)

    # Bin selection
    if K is None:
        K = _auto_K(n_phases, max_len_obs, q=q_null, min_tail_exp=5.0)
    K = max(3, K)

    # Observed histogram: bins 2..K-1 plus tail >=K
    obs_counts = np.zeros(K - 2 + 1, dtype=int)
    for m in lengths:
        if m >= K:
            obs_counts[-1] += 1
        elif m >= 2:
            obs_counts[m - 2] += 1

    # Theoretical expected counts
    pmf = _expected_pmf_shifted_geometric(K, q=q_null)
    exp_counts = pmf * n_phases

    # Chi-square distance (we use permutations for p-value, not chi-square asymptotics)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat_obs = np.nansum((obs_counts - exp_counts) ** 2 / np.where(exp_counts > 0, exp_counts, np.nan))

    # Permutation distribution
    stats_perm = np.empty(B, dtype=float)
    for b in range(B):
        x_perm = rng.permutation(x_arr)
        lens_b = monotonic_phase_lengths(x_perm, drop_ties=drop_ties)

        counts_b = np.zeros_like(obs_counts)
        for m in lens_b:
            if m >= K:
                counts_b[-1] += 1
            elif m >= 2:
                counts_b[m - 2] += 1

        exp_b = pmf * max(1, len(lens_b))  # adjust expected counts to permuted number of phases
        with np.errstate(divide="ignore", invalid="ignore"):
            stats_perm[b] = np.nansum((counts_b - exp_b) ** 2 / np.where(exp_b > 0, exp_b, np.nan))

    # Permutation p-value (right tail: larger distance = greater deviation from null)
    p_perm = (1.0 + np.sum(stats_perm >= stat_obs)) / (B + 1.0)

    bin_labels = [str(m) for m in range(2, K)] + [f"≥{K}"]

    return PhaseLengthTestResult(
        n=n,
        n_effective=n_eff,
        n_phases=n_phases,
        bins=bin_labels,
        observed_counts=obs_counts,
        expected_counts=exp_counts,
        statistic=float(stat_obs),
        p_perm=float(p_perm),
        B=B,
        q_null=q_null,
        seed=seed,
    )

# ---------------------------------------------------------------------
# Optional: quick CLI demo when this file is executed directly
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Permutation test for monotonic phase-length distribution vs. i.i.d."
    )
    parser.add_argument("--n", type=int, default=500, help="Length of a synthetic series")
    parser.add_argument("--ar1", type=float, default=None, help="If set, simulate AR(1) with this phi; else iid N(0,1)")
    parser.add_argument("--B", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.ar1 is None:
        x = rng.normal(size=args.n)
        label = "iid N(0,1)"
    else:
        phi = float(args.ar1)
        eps = rng.normal(size=args.n)
        x = np.empty_like(eps)
        x[0] = eps[0]
        for t in range(1, len(eps)):
            x[t] = phi * x[t - 1] + eps[t]
        label = f"AR(1) phi={phi:g}"

    res = phase_length_distribution_test(x, B=args.B, seed=args.seed)
    print(f"Series: {label}")
    print(f"n={res.n}, n_effective={res.n_effective}, phases={res.n_phases}")
    print("Bins:", ", ".join(res.bins))
    print("Observed:", res.observed_counts.tolist())
    print("Expected:", [round(v, 2) for v in res.expected_counts.tolist()])
    print(f"Statistic: {res.statistic:.4f}")
    print(f"Permutation p-value: {res.p_perm:.4g} (B={res.B})")