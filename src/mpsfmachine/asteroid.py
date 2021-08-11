import fitsio
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import tess_cloud as tc
import pandas as pd

from dataclasses import dataclass
from astropy.io import fits
import numpy.typing as npt
from scipy import sparse

from .utils import _make_A_polar
from scipy.interpolate import interp2d


@dataclass
class Asteroid(object):
    time: npt.NDArray
    flux: npt.NDArray
    flux_err: npt.NDArray
    ccd: npt.NDArray
    camera: npt.NDArray
    sector: npt.NDArray
    col: npt.NDArray
    row: npt.NDArray
    Col: npt.NDArray
    Row: npt.NDArray
    nsubtimes: int = 3

    def __post_init__(self):
        dt = np.median(np.diff(self.time)) / 2
        self.subtime = np.hstack(
            [np.linspace(t - dt, t + dt, self.nsubtimes + 1)[:-1] for t in self.time]
        )
        self.subcol = np.interp(self.subtime, self.time, self.col)
        self.subrow = np.interp(self.subtime, self.time, self.row)
        self.dx, self.dy = (self.Col - self.subcol[:, None, None]), (
            self.Row - self.subrow[:, None, None]
        )

        self.phi, self.r = np.arctan2(self.dy, self.dx), np.hypot(self.dx, self.dy)

    def __repr__(self):
        return "Asteroid"

    @property
    def shape(self):
        return self.flux.shape

    def __getitem__(self, item):
        raise ValueError("broken")
        return Asteroid(
            self.time[item],
            self.flux[item],
            self.flux_err[item],
            self.ccd[item],
            self.camera[item],
            self.sector[item],
            self.col[item],
            self.row[item],
            Col=self.Col[item],
            Row=self.Row[item],
            nsubtimes=self.nsubtimes,
        )

    @staticmethod
    def read(filename: str, nsubtimes: int = 3):
        hdu = fits.open(filename)
        shape = hdu[1].data["FLUX"].shape
        Y, X = np.mgrid[: shape[1], : shape[2]].astype(float)
        corner_col, corner_row = (
            hdu[1].data["CORNER_COLUMN"].astype(float),
            hdu[1].data["CORNER_ROW"].astype(float),
        )
        X = (
            X[None, :, :]
            + np.meshgrid(np.ones(nsubtimes), corner_col,)[
                1
            ].ravel()[:, None, None]
        )
        Y = (
            Y[None, :, :]
            + np.meshgrid(np.ones(nsubtimes), corner_row,)[
                1
            ].ravel()[:, None, None]
        )
        return Asteroid(
            hdu[1].data["TIME"],
            hdu[1].data["FLUX"],
            hdu[1].data["FLUX_ERR"],
            hdu[1].data["CCD"],
            hdu[1].data["CAMERA"],
            hdu[1].data["SECTOR"],
            hdu[1].data["TARGET_COLUMN"],
            hdu[1].data["TARGET_ROW"],
            Col=X,
            Row=Y,
            nsubtimes=nsubtimes,
        )

    def _get_prf(self, plot=False):
        rr = np.linspace(0, 2048, 5, dtype=int) + 1
        rr[-2:] -= 1
        cc = np.linspace(0, 2048, 5, dtype=int) + 45
        cc[-2:] -= 1

        jdx = int(np.median(np.nanargmin(np.abs((rr[:, None] - self.row)), axis=0)))
        idx = int(np.median(np.nanargmin(np.abs((cc[:, None] - self.col)), axis=0)))
        fname = f"https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0001/cam{self.camera[0]}_ccd{self.ccd[0]}/tess2018243163600-prf-{self.camera[0]}-{self.ccd[0]}-row{rr[jdx]:04}-col{cc[idx]:04}.fits"

        hdu = fits.open(fname)
        self.prfhdr = hdu[0].header
        self.prf = fitsio.read(hdu._file.name)
        crval1p, crval2p, cdelt1p, cdelt2p = (
            self.prfhdr["CRVAL1P"],
            self.prfhdr["CRVAL2P"],
            self.prfhdr["CDELT1P"],
            self.prfhdr["CDELT2P"],
        )
        col = np.arange(0.5, np.shape(self.prf)[1] + 0.5)
        row = np.arange(0.5, np.shape(self.prf)[0] + 0.5)
        col = (col - np.size(col) / 2) * cdelt1p
        row = (row - np.size(row) / 2) * cdelt2p
        # row, col = int(self.prfhdr["CRVAL2P"]), int(self.prfhdr["CRVAL1P"])
        # col = (
        #     np.arange(0, 0 + self.prf.shape[0]) * self.prfhdr["CDELT1P"]
        # ) - self.prfhdr["CDELT1P"] * self.prf.shape[0] / 2
        # row = (
        #     np.arange(0, 0 + self.prf.shape[1]) * self.prfhdr["CDELT2P"]
        # ) - self.prfhdr["CDELT2P"] * self.prf.shape[1] / 2

        def get_LFD(plot=plot):
            row, col = np.meshgrid(col, row)
            phi, r = np.arctan2(row, col), np.hypot(col, row)
            X = _make_A_polar(phi.ravel(), r.ravel())
            self.X = _make_A_polar(self.phi.ravel(), self.r.ravel())
            self.w = self._solve_linear_model(
                X,
                (
                    (np.log10(self.prf) + 2.50000) * (np.log10(self.prf) > -2.50000)
                ).ravel(),
                prior_mu=np.zeros(X.shape[1]),
                prior_sigma=np.ones(X.shape[1]) * 1e10,
            )
            self.gw1 = self.prfhdr["CDELT1P"] * self._solve_linear_model(
                X,
                np.gradient(self.prf, axis=0).ravel(),
                prior_mu=np.zeros(X.shape[1]),
                prior_sigma=np.ones(X.shape[1]) * 1e10,
            )
            self.gw2 = self.prfhdr["CDELT1P"] * self._solve_linear_model(
                X,
                np.gradient(self.prf, axis=1).ravel(),
                prior_mu=np.zeros(X.shape[1]),
                prior_sigma=np.ones(X.shape[1]) * 1e10,
            )

            self.fmodel = 10 ** (self.X.dot(self.w) - 2.50000)
            self.fmodel -= np.min(self.fmodel)
            self.gmodel1 = self.X.dot(self.gw1)
            self.gmodel2 = self.X.dot(self.gw2)

            if plot:
                with plt.style.context("seaborn-white"):
                    fig, ax = plt.subplots(
                        3, 2, figsize=(10, 12), sharex=True, sharey=True
                    )
                    im = ax[0, 0].pcolormesh(
                        col, row, self.prf, shading="auto", cmap="Greys_r"
                    )
                    plt.colorbar(im, ax=ax[0, 0])
                    im = ax[0, 1].pcolormesh(
                        col,
                        row,
                        10 ** (X.dot(self.w).reshape(col.shape) - 2.50000),
                        shading="auto",
                        cmap="Greys_r",
                    )
                    plt.colorbar(im, ax=ax[0, 1])

                    im = ax[1, 0].pcolormesh(
                        col,
                        row,
                        np.gradient(self.prf, axis=0),
                        cmap="coolwarm",
                        shading="auto",
                    )
                    plt.colorbar(im, ax=ax[1, 0])
                    im = ax[1, 1].pcolormesh(
                        col,
                        row,
                        X.dot(self.gw1).reshape(self.prf.shape),
                        cmap="coolwarm",
                        shading="auto",
                    )
                    plt.colorbar(im, ax=ax[1, 1])

                    im = ax[2, 0].pcolormesh(
                        col,
                        row,
                        np.gradient(self.prf, axis=1),
                        cmap="coolwarm",
                        shading="auto",
                    )
                    plt.colorbar(im, ax=ax[2, 0])
                    im = ax[2, 1].pcolormesh(
                        col,
                        row,
                        X.dot(self.gw2).reshape(self.prf.shape),
                        cmap="coolwarm",
                        shading="auto",
                    )
                    plt.colorbar(im, ax=ax[2, 1])

                    ax[0, 0].set(title="Data", ylabel="$\delta$ Row")
                    ax[0, 1].set(title="Model")
                    ax[1, 0].set(ylabel="$\delta$ Row")
                    ax[2, 0].set(ylabel="$\delta$ Row", xlabel="$\delta$ Column")
                    ax[2, 1].set(ylabel="$\delta$ Row", xlabel="$\delta$ Column")
                    plt.subplots_adjust(hspace=0.1, wspace=0.0)
                    return fig

        def get_simple_interp():
            f = interp2d(
                col, row, (np.log10(self.prf) + 2.5) * (np.log10(self.prf) > -2.5)
            )
            self.fmodel = (
                np.asarray(
                    [
                        10 ** (f(self.dx[idx][0], self.dy[idx][:, 0]) - 2.5)
                        for idx in range(self.shape[0] * self.nsubtimes)
                    ]
                ).ravel()
                / self.nsubtimes
            )
            self.fmodel -= np.min(self.fmodel)

            f = interp2d(col, row, np.gradient(self.prf, axis=0))
            self.gmodel1 = (
                self.prfhdr["CDELT1P"]
                * np.asarray(
                    [
                        f(self.dx[idx][0], self.dy[idx][:, 0])
                        for idx in range(self.shape[0] * self.nsubtimes)
                    ]
                ).ravel()
                / self.nsubtimes
            )
            f = interp2d(col, row, np.gradient(self.prf, axis=1))
            self.gmodel2 = (
                self.prfhdr["CDELT2P"]
                * np.asarray(
                    [
                        f(self.dx[idx][0], self.dy[idx][:, 0])
                        for idx in range(self.shape[0] * self.nsubtimes)
                    ]
                ).ravel()
                / self.nsubtimes
            )

        def bin_down_subtimes():
            self.fmodel = np.sum(
                [
                    self.fmodel.reshape(self.dx.shape)[idx :: self.nsubtimes]
                    for idx in range(self.nsubtimes)
                ],
                axis=0,
            ).ravel()
            self.gmodel1 = np.sum(
                [
                    self.gmodel1.reshape(self.dx.shape)[idx :: self.nsubtimes]
                    for idx in range(self.nsubtimes)
                ],
                axis=0,
            ).ravel()
            self.gmodel2 = np.sum(
                [
                    self.gmodel2.reshape(self.dx.shape)[idx :: self.nsubtimes]
                    for idx in range(self.nsubtimes)
                ],
                axis=0,
            ).ravel()

        get_simple_interp()
        bin_down_subtimes()

    @property
    def _A_single_flux_all_bkg(self):
        """Makes a design matrix for the asteroid that has a single flux value."""
        npix = self.shape[1] * self.shape[2]
        ntime = self.shape[0]
        Abkg = sparse.lil_matrix((ntime * npix, ntime))
        for idx in range(ntime):
            Abkg[npix * idx : npix * (idx + 1), idx] = 1
        t = (self.time - self.time.mean()) / (self.time.max() - self.time.min())
        t = (t[:, None, None] * np.ones(self.shape)).ravel()
        A = np.vstack(
            [
                self.gmodel1,
                self.gmodel2,
                self.gmodel1 * self.gmodel2,
                self.gmodel1 * t,
                self.gmodel2 * t,
                self.gmodel1 * self.gmodel2 * t,
                self.gmodel1 * t ** 2,
                self.gmodel2 * t ** 2,
                self.gmodel1 * self.gmodel2 * t ** 2,
                # self.gmodel1 ** 2,
                # self.gmodel2 ** 2,
                # self.gmodel1 * t ** 3,
                # self.gmodel2 * t ** 3,
                # self.gmodel1 * self.gmodel2 * t ** 3,
                # self.gmodel1 ** 2 * self.gmodel2,
                # self.gmodel1 * self.gmodel2 ** 2,
                # self.gmodel1 ** 2 * self.gmodel2 ** 2,
            ]
        ).T
        return sparse.hstack(
            [
                sparse.csr_matrix(self.fmodel[:, None]),
                Abkg,
                sparse.csr_matrix(A),
            ],
            format="csr",
        )

    @property
    def _A_single_flux(self):
        """Makes a design matrix for the asteroid that has a single flux value."""
        t = (self.time - self.time.mean()) / (self.time.max() - self.time.min())
        t = (t[:, None, None] * np.ones(self.shape)).ravel()
        return np.vstack(
            [
                self.fmodel,
                # self.gmodel1,
                # self.gmodel2,
                # self.gmodel1 * self.gmodel2,
                # self.gmodel1 * t,
                # self.gmodel2 * t,
                # self.gmodel1 * self.gmodel2 * t,
                # self.gmodel1 * t ** 2,
                # self.gmodel2 * t ** 2,
                # self.gmodel1 * self.gmodel2 * t ** 2,
                # self.gmodel1 ** 2,
                # self.gmodel2 ** 2,
                # self.gmodel1 * t ** 3,
                # self.gmodel2 * t ** 3,
                # self.gmodel1 * self.gmodel2 * t ** 3,
                # self.gmodel1 ** 2 * self.gmodel2,
                # self.gmodel1 * self.gmodel2 ** 2,
                # self.gmodel1 ** 2 * self.gmodel2 ** 2,
                np.ones(np.product(self.shape))[None, :],
            ]
        ).T

    @property
    def _A_all_flux(self):
        """Makes a design matrix for the asteroid that has a single flux value"""
        npix = self.shape[1] * self.shape[2]
        ntime = self.shape[0]
        A1 = sparse.lil_matrix((ntime * npix, ntime))
        A2 = sparse.lil_matrix((ntime * npix, ntime))
        for idx in range(ntime):
            A1[npix * idx : npix * (idx + 1), idx] = self.fmodel[
                npix * idx : npix * (idx + 1)
            ]
            A2[npix * idx : npix * (idx + 1), idx] = 1
        t = (self.time - self.time.mean()) / (self.time.max() - self.time.min())
        t = (t[:, None, None] * np.ones(self.shape)).ravel()
        return sparse.hstack(
            [
                A1,
                A2,
                sparse.lil_matrix(
                    np.vstack(
                        [
                            self.gmodel1,
                            self.gmodel2,
                            self.gmodel1 * self.gmodel2,
                            self.gmodel1 * t,
                            self.gmodel2 * t,
                            self.gmodel1 * self.gmodel2 * t,
                            self.gmodel1 * t ** 2,
                            self.gmodel2 * t ** 2,
                            # self.gmodel1 * self.gmodel2 * t ** 2,
                            # self.gmodel1 * t ** 3,
                            # self.gmodel2 * t ** 3,
                            # self.gmodel1 * self.gmodel2 * t ** 3,
                            # self.gmodel1 ** 2,
                            # self.gmodel2 ** 2,
                            # self.gmodel1 ** 2 * self.gmodel2,
                            # self.gmodel1 * self.gmodel2 ** 2,
                            # self.gmodel1 ** 2 * self.gmodel2 ** 2,
                        ]
                    ).T
                ),
            ],
            format="csr",
        )

    @staticmethod
    def _solve_linear_model(
        A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None, errors=False
    ):
        """
                Solves a linear model with design matrix A and observations y:
                    Aw = y
                return the solutions w for the system assuming Gaussian priors.
                Alternatively the observation errors, priors, and a boolean mask for the
                observations (row axis) can be provided.
                Adapted from Luger, Foreman-Mackey & Hogg, 2017
                (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)
                Parameters
                ----------
                A: numpy ndarray or scipy sparce csr matrix
                    Desging matrix with solution basis
                    shape n_observations x n_basis
                y: numpy ndarray
                    Observations
                    shape n_observations
                y_err: numpy ndarray, optional
                    Observation errors
                    shape n_observations
                prior_mu: float, optional
                    Mean of Gaussian prior values for the weights (w)
                prior_sigma: float, optional
                    Standard deviation of Gaussian prior values for the weights (w)
                k: boolean, numpy ndarray, optional
                    Mask that sets the observations to be used to solve the system
                    shape n_observations
                errors: boolean
                    Whether to return error estimates of the best fitting weights
                Returns
                -------
                w: numpy ndarray
                    Array with the estimations for the weights
                    shape n_basis
                werrs: numpy ndarray
                    Array with the error estimations for the weights, returned if `error`
        is True
                    shape n_basis
        """
        if k is None:
            k = np.ones(len(y), dtype=bool)

        if y_err is not None:
            sigma_w_inv = A[k].T.dot(A[k].multiply(1 / y_err[k, None] ** 2))
            B = A[k].T.dot((y[k] / y_err[k] ** 2))
        else:
            sigma_w_inv = A[k].T.dot(A[k])
            B = A[k].T.dot(y[k])

        if prior_mu is not None and prior_sigma is not None:
            sigma_w_inv += np.diag(1 / prior_sigma ** 2)
            B += prior_mu / prior_sigma ** 2

        if isinstance(sigma_w_inv, (sparse.csr_matrix, sparse.csc_matrix, np.matrix)):
            sigma_w_inv = np.asarray(sigma_w_inv)

        w = np.linalg.solve(sigma_w_inv, B)
        if errors is True:
            w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            return w, w_err
        return w
