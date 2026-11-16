from aad import tape
from main import bs_analytic
import pandas as pd
from dataclasses import dataclass


def aad_greeks(spot, strike, vol, rate, ttm):
    # fresh tape and leaves per call: tape.clear() does not reset leaf adjoints
    t = tape([])
    leaves = [t.new_leaf(x) for x in (spot, strike, vol, rate, ttm)]
    pv, delta, vega = bs_analytic(t, *leaves)
    rho = leaves[3].node.adjoint
    dpv_dttm = leaves[4].node.adjoint
    return pv, delta, vega, rho, dpv_dttm


@dataclass
class BSRisk:
    spot: float
    strike: float
    vol: float
    rate: float
    ttm: float
    pv: float
    delta: float
    vega: float
    rho: float
    theta: float  # dPV/dt per year, i.e. -dPV/dttm
    gamma: float
    vanna: float
    volga: float

    def pnl_explain(self, ds, dv, dr, dt=1 / 252) -> pd.Series:
        terms = {
            "delta": self.delta * ds,
            "gamma": 0.5 * self.gamma * ds**2,
            "vega": self.vega * dv,
            "volga": 0.5 * self.volga * dv**2,
            "vanna": self.vanna * ds * dv,
            "rho": self.rho * dr,
            "theta": self.theta * dt,
        }
        out = pd.Series(terms)
        out["total"] = out.sum()
        return out

    def pnl(self, ds, dv, dr, dt=1 / 252) -> float:
        pv_new = aad_greeks(
            self.spot + ds, self.strike, self.vol + dv, self.rate + dr, self.ttm - dt
        )[0]
        return pv_new - self.pv


def bs_risk(spot, strike, vol, rate, ttm, spot_bump=0.01, vol_bump=0.01) -> BSRisk:
    pv, delta, vega, rho, dpv_dttm = aad_greeks(spot, strike, vol, rate, ttm)

    # second order: central differences of the AAD delta and vega
    # to get gamma, vanna and volgamma

    hs = spot * spot_bump
    _, d_sup, _, _, _ = aad_greeks(spot + hs, strike, vol, rate, ttm)
    _, d_sdwn, _, _, _ = aad_greeks(spot - hs, strike, vol, rate, ttm)
    gamma = (d_sup - d_sdwn) / (2 * hs)

    hv = vol * vol_bump
    _, d_vup, v_vup, _, _ = aad_greeks(spot, strike, vol + hv, rate, ttm)
    _, d_vdwn, v_vdwn, _, _ = aad_greeks(spot, strike, vol - hv, rate, ttm)
    vanna = (d_vup - d_vdwn) / (2 * hv)
    volga = (v_vup - v_vdwn) / (2 * hv)

    return BSRisk(
        spot,
        strike,
        vol,
        rate,
        ttm,
        pv,
        delta,
        vega,
        rho,
        -dpv_dttm,
        gamma,
        vanna,
        volga,
    )
