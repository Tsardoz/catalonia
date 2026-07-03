"""
B-spline basis utilities for scalar-on-function regression.
"""
import numpy as np
from scipy.interpolate import BSpline


def build_bspline_basis(grid: np.ndarray, n_basis: int = 5,
                        degree: int = 3) -> np.ndarray:
    """
    Build a cubic B-spline basis matrix evaluated on a grid.

    Parameters
    ----------
    grid : 1-D array of evaluation points (e.g. calendar DOY grid).
    n_basis : number of basis functions.
    degree : spline degree (3 = cubic).

    Returns
    -------
    B : array of shape (len(grid), n_basis).
    """
    # Place interior knots at quantiles of the grid
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        raise ValueError(
            f"n_basis={n_basis} too small for degree={degree}: "
            f"need at least {degree + 1}"
        )

    if n_interior == 0:
        interior_knots = np.array([])
    else:
        quantiles = np.linspace(0, 100, n_interior + 2)[1:-1]
        interior_knots = np.percentile(grid, quantiles)

    # Full knot vector with boundary knots repeated (degree + 1) times
    lo, hi = grid[0], grid[-1]
    knots = np.concatenate([
        np.repeat(lo, degree + 1),
        interior_knots,
        np.repeat(hi, degree + 1),
    ])

    # Evaluate each basis function
    B = np.zeros((len(grid), n_basis))
    for j in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[j] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=False)
        B[:, j] = spl(grid)

    # Replace NaN at boundaries with 0
    B = np.nan_to_num(B, nan=0.0)

    return B
