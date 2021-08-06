""" Collection of utility functions"""
import numpy as np
import pandas as pd
import functools

from scipy import sparse
from patsy import dmatrix
from psfmachine.utils import wrapped_spline


def _make_A_polar(phi, r, rmin=0.5, rmax=9, n_r_knots=10, n_phi_knots=10):
    # create the spline bases for radius and angle
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=n_phi_knots).T)
    r_knots = np.linspace(rmin ** 0.5, rmax ** 0.5, n_r_knots) ** 2
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(np.hstack([r, rmin, rmax])), "knots": r_knots},
            )
        )
    )[:-2]

    # build full desing matrix
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    return X
