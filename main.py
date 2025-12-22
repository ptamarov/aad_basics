from aad import new_leaf, TAPE
import numpy as np
import seaborn as sns
import pandas as pd
from aad import number

sns.set_style("whitegrid")


def my_plot(df: pd.DataFrame, x: str, y: str):
    sns.scatterplot(  # type: ignore
        data=df,
        x=x,
        y=y,
        marker="o",
        facecolor="none",
        edgecolor="k",
        s=25,
    )
    sns.lineplot(  # type: ignore
        data=df,
        x=x,
        y=y,
        color="k",
        linewidth=0.75,
    )


def bs_analytic(
    spot: number, strike: number, vol: number, rate: number, ttm: number
) -> list[float]:

    df = (-rate * ttm).exp()
    fwd = spot / df
    z = fwd / strike
    vvol = ttm.sqrt() * vol
    z0 = z.log() / vvol
    hvol = vvol / 2
    z1 = z0 + hvol
    z2 = z0 - hvol
    out = (fwd * z1.norm_cdf() - strike * z2.norm_cdf()) * df
    pv = out.value

    TAPE.seed()
    TAPE.backprop()
    TAPE.clear()

    delta = spot.node.n.adjoint.v
    vega = vol.node.n.adjoint.v

    # could also get Rho or Theta

    return [pv, delta, vega]


def bs_montecarlo(
    S: float,
    K: float,
    bsvol: float,
    rate: float,
    num_paths: int,
    T: float,
    seed: int = 12,
):

    payoffs: list[float] = []
    deltas: list[float] = []
    vegas: list[float] = []

    rng = np.random.default_rng(seed)

    for _ in range(num_paths):

        TAPE.clear()  # clear the tape for each path
        spot = new_leaf(S)
        vol = new_leaf(bsvol)
        drift = new_leaf(rate) - 0.50 * vol * vol  # forward
        logspot = spot.log()
        sqrtT = float(np.sqrt(T))
        noise = rng.normal()
        logspot = logspot + T * drift + vol * sqrtT * noise
        endval = logspot.exp()
        payoff = (endval - K).max(0.00)
        payoffs.append(payoff.value)

        TAPE.seed()
        TAPE.backprop()

        delta = spot.node.n.adjoint
        vega = vol.node.n.adjoint

        deltas.append(delta.v)
        vegas.append(vega.v)

    delta = sum(deltas) / num_paths
    pv = sum(payoffs) / num_paths
    vega = sum(vegas) / num_paths

    return pv, delta, vega
