from math import sqrt, erfc
from typing import Sequence, NamedTuple, Optional

class TurningPointTestResult(NamedTuple):
    n: int                 # original length (including any ties)
    n_effective: int       # length after optional tie handling
    T: int                 # observed number of turning points
    mu: float              # expected value of T under H0
    var: float             # variance of T under H0
    z: float               # z statistic (normal approximation)
    p_value: float         # two-sided p-value

def turning_point_test(
    x: Sequence[float],
    *,
    drop_ties: bool = True,
    continuity_correction: bool = True
) -> TurningPointTestResult:
    """
    Classical Turning Point Test for independence of a (continuous) time series.

    Parameters
    ----------
    x : Sequence[float]
        Time series values (assumed from a continuous distribution under H0).
    drop_ties : bool, default True
        If True, compresses consecutive equal values (flat runs) so that the
        turning-point count follows the classical definition (no ties).
        If False and ties exist, the test can become conservative/ill-defined.
    continuity_correction : bool, default True
        Apply a 0.5 continuity correction to the normal approximation.

    Returns
    -------
    TurningPointTestResult
        Named tuple with test details.

    Notes
    -----
    - A turning point is an interior index i with either
      x[i-1] < x[i] > x[i+1] (local max) or x[i-1] > x[i] < x[i+1] (local min).
    - Under H0 (iid continuous), T ~ approx Normal(mu, var) for large n, where
        mu  = (2n - 4) / 3
        var = (16n - 29) / 90
      (Bienaymé; Kendall & Stuart).  # See: Wikipedia summary.  [CITATION]
    - The test is relatively powerful against cyclicity, but poor against trend.
      Consider complementing with runs tests or Ljung–Box for serial dependence.
    """
    if not x:
        raise ValueError("Input series x must be non-empty.")
    arr = list(x)

    # Handle ties (classical test assumes a continuous distribution ⇒ no equal adjacents).
    if drop_ties:
        comp = [arr[0]]
        for v in arr[1:]:
            if v != comp[-1]:
                comp.append(v)
        arr = comp

    n_eff = len(arr)
    if n_eff < 3:
        raise ValueError("Need at least 3 (effective) observations to compute turning points.")

    # Count turning points
    T = 0
    for i in range(1, n_eff - 1):
        if (arr[i-1] < arr[i] > arr[i+1]) or (arr[i-1] > arr[i] < arr[i+1]):
            T += 1

    # Mean and variance under H0 (large-sample approximation)
    mu = (2 * n_eff - 4) / 3.0        # [1](https://en.wikipedia.org/wiki/Turning_point_test)
    var = (16 * n_eff - 29) / 90.0    # [1](https://en.wikipedia.org/wiki/Turning_point_test)

    # z statistic with optional continuity correction
    if var <= 0:
        # Should not occur for n_eff >= 3, but guard anyway.
        z = float("nan")
        p = float("nan")
    else:
        delta = 0.5 if (continuity_correction and (T != mu)) else 0.0
        sign = 1.0 if (T - mu) > 0 else (-1.0 if (T - mu) < 0 else 0.0)
        z = (T - mu - sign * delta) / sqrt(var)

        # Two-sided p-value using the complementary error function (no SciPy needed)
        def norm_sf(xval: float) -> float:
            return 0.5 * erfc(xval / sqrt(2.0))
        p = 2.0 * norm_sf(abs(z))

    return TurningPointTestResult(
        n=len(x),
        n_effective=n_eff,
        T=T,
        mu=mu,
        var=var,
        z=z,
        p_value=p
    )

if __name__ == "__main__":
    # Example quick test when running as a standalone script
    import random
    x = [random.gauss(0, 1) for _ in range(100)]
    result = turning_point_test(x)
    print(result)
