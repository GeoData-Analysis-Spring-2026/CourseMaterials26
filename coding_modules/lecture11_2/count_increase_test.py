from typing import Iterable, Literal, NamedTuple, Optional
import math

class CountIncreasesResult(NamedTuple):
    n: int                 # original length
    n_trials: int          # comparisons used (<= n-1 if ties excluded)
    n_increases: int       # c = #{x_{i+1} > x_i}
    n_decreases: int       # #{x_{i+1} < x_i}
    n_ties: int            # #{x_{i+1} = x_i}
    alternative: str       # 'two-sided' | 'increasing' | 'decreasing'
    p_value: float

def _binom_pmf(i: int, n: int, p: float = 0.5) -> float:
    """Exact Binomial PMF using log-gamma for stability."""
    if i < 0 or i > n:
        return 0.0
    logc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
    return math.exp(logc + i * math.log(p) + (n - i) * math.log(1 - p))

def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """Survival function P[X >= k] for X~Bin(n,p), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # Sum from k..n; for p=0.5 and moderate n this is fine; stable via PMF recurrence if desired.
    return sum(_binom_pmf(i, n, p) for i in range(k, n + 1))

def count_increases_test(
    x: Iterable[float],
    *,
    alternative: Literal["two-sided", "increasing", "decreasing"] = "two-sided",
    include_ties_as: Optional[Literal["increase", "decrease", "random"]] = None
) -> CountIncreasesResult:
    """
    Count-Increases test for trend based on adjacent comparisons.

    Parameters
    ----------
    x : Iterable[float]
        Time-ordered series, length n >= 2.
    alternative : {'two-sided', 'increasing', 'decreasing'}, default 'two-sided'
        - 'increasing': right-tail test, P[C >= c]
        - 'decreasing': left-tail test, P[C <= c]
        - 'two-sided': 2 * min{ right-tail, left-tail } (clipped at 1)
    include_ties_as : {'increase','decrease','random', None}, default None
        How to handle ties (x_{i+1} == x_i):
          - None      : exclude ties (reduce trial count; recommended)
          - 'increase': count all ties as increases
          - 'decrease': count all ties as decreases
          - 'random'  : break ties by fair coin (adds expected 0.5 per tie)

    Returns
    -------
    CountIncreasesResult
        Summary and exact p-value under H0: i.i.d., P(increase)=P(decrease)=1/2.

    Notes
    -----
    Under H0 (i.i.d. continuous), ties occur with probability ~0, so excluding ties
    mirrors the ideal setting. If your data are quantized, 'random' can be used to
    avoid systematic bias from ties.
    """
    x_list = list(x)
    n = len(x_list)
    if n < 2:
        raise ValueError("Need at least two observations.")

    inc = dec = ties = 0
    for i in range(n - 1):
        d = x_list[i + 1] - x_list[i]
        if d > 0:
            inc += 1
        elif d < 0:
            dec += 1
        else:
            ties += 1

    # Optionally fold ties into increases/decreases or random coin flips
    if include_ties_as == "increase":
        inc += ties
        n_trials = (n - 1)
    elif include_ties_as == "decrease":
        dec += ties
        n_trials = (n - 1)
    elif include_ties_as == "random":
        # add Binomial(ties, 0.5) expected value by sampling a fair coin
        # (alternatively, you could add ties/2 deterministically for an expected-p variant)
        rnd_in_ties = sum((1 if (hash((i, x_list[i])) & 1) else 0) for i in range(n - 1) if x_list[i + 1] == x_list[i])
        inc += rnd_in_ties
        dec += ties - rnd_in_ties
        n_trials = (n - 1)
    else:
        # exclude ties entirely
        n_trials = inc + dec

    if n_trials == 0:
        # all pairs tied → no information
        return CountIncreasesResult(
            n=n, n_trials=0, n_increases=inc, n_decreases=dec, n_ties=ties,
            alternative=alternative, p_value=float("nan")
        )

    # Exact p-values with p=0.5
    if alternative == "increasing":
        # right tail: P[C >= c]
        p_val = _binom_sf(inc, n_trials, 0.5)
    elif alternative == "decreasing":
        # left tail: P[C <= c] = P[C >= n_trials - c] by symmetry
        p_val = _binom_sf(n_trials - inc, n_trials, 0.5)
    else:  # two-sided
        right = _binom_sf(inc, n_trials, 0.5)
        left  = _binom_sf(n_trials - inc, n_trials, 0.5)
        p_val = min(1.0, 2.0 * min(right, left))

    return CountIncreasesResult(
        n=n,
        n_trials=n_trials,
        n_increases=inc,
        n_decreases=dec,
        n_ties=ties,
        alternative=alternative,
        p_value=p_val,
    )

if __name__ == "__main__":
    # Example quick test when running as a standalone script
    import random
    x = [random.gauss(0, 1) for _ in range(100)]
    result = count_increases_test(x)
    print(result)
