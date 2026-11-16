import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, OptimizeResult  # type: ignore
from aad import tape
from svi_ift import jacobian_svi, amat_solve

array = NDArray[np.float64]


class SliceSVI:
    def __init__(self, t: float) -> None:
        self.ttm = t
        self.params: array = np.asarray([0.001, 0.001, -0.70, 0.00, 1.00])
        self.x = np.array(0)
        self.y = np.array(0)
        self.residuals = np.array(0)
        self.is_fitted = False
        self.tape = tape([])
        self.pricing_results: dict[str, float | array] = {}

    def bstermvar(self, z: array) -> array:
        a, b, r, m, s = self.params
        z = z - m
        v = a + b * (r * z + np.sqrt(z**2 + s**2))
        return v

    def bsvol(self, z: array) -> array:
        return np.sqrt(self.bstermvar(z) / self.ttm)

    def fit(self, zs: array, ivs: array, guess: array | None = None) -> None:
        self.x = np.asarray(zs)
        y = self.ttm * np.asarray(ivs) ** 2
        self.y = y

        guess = np.asarray(guess) if guess is not None else self.params

        def f(z: array, P: array):
            a, b, r, m, s = P
            z = z - m
            return a + b * (r * z + np.sqrt(z**2 + s**2))

        def residuals(P: array) -> array:
            return f(self.x, P) - self.y

        result: OptimizeResult = least_squares(  # type: ignore
            residuals,
            guess,
            jac="2-point",
            method="trf",
            # no lower bound on a: admissibility is checked after the fit
            bounds=(
                (-np.inf, 1e-4, -1, -np.inf, 1e-4),
                (np.inf, np.inf, 1, np.inf, 100),
            ),
        )

        self.fit_result = result  # type: ignore
        self.params = np.array(result.x, dtype=float)  # type: ignore
        self.is_fitted = True
        self.residuals = residuals(self.params)

        # minimum total variance of the slice, a + b s sqrt(1 - rho^2), must be >= 0
        a, b, r, _, s = self.params
        self.min_variance = a + b * s * np.sqrt(1 - r**2)
        if self.min_variance < 0:
            warnings.warn(f"SVI fit has negative minimum variance {self.min_variance:.3e}")

    # we compute bs_price but use the IVs provided by the SliceSVI.

    def bscall(
        self,
        S0: float,
        K: float,
        r0: float,
        tau: float,
        compute_aad_greeks: bool = True,
    ) -> None:

        if self.is_fitted is False and compute_aad_greeks:
            raise ValueError("cannot compute aad Greeks on non-fitted slice!")

        if not compute_aad_greeks:
            raise NotImplementedError()

        the_tape = tape([])
        rate = the_tape.new_leaf(r0)
        spot = the_tape.new_leaf(S0)

        # instrumented SVI params
        params_aad = [the_tape.new_leaf(p) for p in self.params]

        ttm = the_tape.new_leaf(tau)
        # compute bs price
        df = (-rate * ttm).exp()
        fwd = spot / df
        z = (fwd / K).log()
        # svi needs y with K = F e^y i.e. y = log(K/F)
        # so feed it -z above
        a, b, r, m, s = params_aad
        z0 = -z - m
        u = z0 * z0 + s * s
        vvol = (a + b * (r * z0 + u.sqrt())).sqrt()
        z0 = z / vvol
        hvol = vvol / 2
        z1 = z0 + hvol
        z2 = z0 - hvol
        out = (fwd * z1.norm_cdf() - K * z2.norm_cdf()) * df
        pv = out.value
        the_tape.seed()
        the_tape.backprop()
        the_tape.clear()
        dg = np.asarray([x.node.adjoint for x in [a, b, r, m, s]])
        x0 = amat_solve(self.residuals, self.x, self.params, dg)
        J = jacobian_svi(self.x, self.params)
        vegas = (
            2 * self.ttm * np.sqrt(self.y / self.ttm) * (J @ x0)
        )  # account for v = T σ^2

        results: dict[str, float | array] = {
            "netvalue": pv,
            "vegas": vegas,
            "delta": spot.node.adjoint,
            "fwd_delta": fwd.node.adjoint,
            "rho": rate.node.adjoint,
            "theta": ttm.node.adjoint,
        }
        self.pricing_results = results
