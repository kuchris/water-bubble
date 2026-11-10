"""Equilibrium shape of a gas bubble floating at a liquid surface.

Solves the axisymmetric Young-Laplace equations for the three interfaces
(submerged wall, thin-film cap, outer meniscus). Everything here is
dimensionless: lengths in capillary lengths l_c = sqrt(gamma / (rho g)),
pressures in gamma / l_c. The shape therefore depends on one number only,
R / l_c (the Bond number is its square), which is how surface tension enters.

Geometry (z up, undisturbed liquid level at z = 0):
  wall     : from the bottom apex up to the junction ring, pressure jump P + z
  cap      : spherical film with two faces, radius 4 / P
  meniscus : outer free surface, decays to z = 0 far away
At the junction ring the film (tension 2*gamma) balances wall + meniscus
(gamma each), which forces all three to be tangent there.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import k0, k1

NW = 128          # wall samples (uniform in z)
NM = 128          # meniscus samples (uniform in r)
MEN_SPAN = 6.0    # meniscus is tabulated out to r_c + MEN_SPAN
_MATCH = 2.0      # beyond r_c + _MATCH the linear K0 tail is used
_CACHE = Path(__file__).with_name("yl_table.npz")


def _wall(b):
    """Integrate the wall from the apex (curvature jump b) until it turns over."""
    def f(s, y):
        r, ze, ph, _ = y
        sp = np.sin(ph)
        return [np.cos(ph), sp, b + ze - sp / r, np.pi * r * r * sp]

    def top(s, y):
        return y[2] - np.pi
    top.terminal = True

    def equator(s, y):
        return y[2] - 0.5 * np.pi

    e = 1e-4 * min(1.0, 2.0 / b)
    y0 = [e, b * e * e / 4, b * e / 2, 0.0]
    sol = solve_ivp(f, (e, 400.0), y0, events=(top, equator), dense_output=True,
                    rtol=1e-10, atol=1e-13, max_step=0.05 * max(1.0, 2.0 / b))
    if not sol.t_events[0].size or not sol.t_events[1].size:
        raise RuntimeError(f"wall integration failed for b={b}")
    return sol.sol, sol.t_events[1][0], sol.t_events[0][0]


def _meniscus(rc, zc, psi, dense=False):
    """Shoot the meniscus outward. Returns +1 if it ends too high, -1 too low."""
    if zc <= 0:
        return -1, None
    if psi <= 1e-12:
        return +1, None

    def f(s, y):
        r, z, p = y
        return [np.cos(p), -np.sin(p), -z - np.sin(p) / r]

    def flat(s, y):
        return y[2]
    flat.terminal = True

    def sunk(s, y):
        return y[1]
    sunk.terminal = True

    sol = solve_ivp(f, (0.0, 60.0), [rc, zc, psi], events=(flat, sunk),
                    dense_output=dense, rtol=1e-10, atol=1e-13)
    if sol.t_events[0].size:
        return +1, sol
    return -1, sol


def _junction(wall, s_eq, s_end, P):
    """Point on the wall where a cap of radius 4/P joins tangentially."""
    def g(s):
        y = wall(s)
        return y[0] - (4.0 / P) * np.sin(y[2])
    if g(s_eq) >= 0:          # cap would exceed a hemisphere
        return None
    return brentq(g, s_eq, s_end, xtol=1e-14)


def solve(b):
    """Full shape for apex curvature jump b. Returns a dict (units of l_c)."""
    wall, s_eq, s_end = _wall(b)
    ze_top = wall(s_end)[1]

    def trial(zb):
        P = b - zb
        sc = _junction(wall, s_eq, s_end, P)
        if sc is None:
            return -1, None
        r, ze, ph, V = wall(sc)
        return _meniscus(r, zb + ze, np.pi - ph)[0], sc

    lo, hi = -ze_top - 0.01, b * (1 - 1e-9)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if trial(mid)[0] > 0:
            hi = mid
        else:
            lo = mid
    zb = 0.5 * (lo + hi)
    P = b - zb
    sc = _junction(wall, s_eq, s_end, P)
    rc, ze_c, ph_c, Vw = wall(sc)
    zc, th = zb + ze_c, np.pi - ph_c
    Rcap = 4.0 / P
    hcap = Rcap * (1 - np.cos(th))
    V = Vw + np.pi * hcap * hcap * (3 * Rcap - hcap) / 3
    R = (3 * V / (4 * np.pi)) ** (1 / 3)

    # wall radius on a uniform z grid
    s = np.linspace(0.0, sc, 4000)[1:]
    ys = wall(np.maximum(s, 1e-9 * sc))
    ze_s = np.concatenate([[0.0], ys[1]])
    r_s = np.concatenate([[0.0], ys[0]])
    wall_r = np.interp(np.linspace(0.0, ze_c, NW), ze_s, r_s)
    rmax = max(r_s.max(), rc)

    # meniscus height on a uniform r grid; K0 tail past the matching radius
    _, msol = _meniscus(rc, zc, th, dense=True)
    sm = np.linspace(0.0, msol.t[-1], 6000)
    ym = msol.sol(sm)
    rm = rc + _MATCH
    zm = np.interp(rm, ym[0], ym[1])
    Q = zm / k0(rm)                       # capillary "charge": z -> Q K0(r)
    rr = rc + np.linspace(0.0, MEN_SPAN, NM)
    men_z = np.where(rr <= rm, np.interp(rr, ym[0], ym[1]), Q * k0(rr))

    return dict(b=b, R=R, P=P, zb=zb, zc=zc, rc=rc, th=th, Rcap=Rcap,
                rmax=rmax, Q=Q, wall_r=wall_r, men_z=men_z)


def build_table(n=56, b_lo=0.012, b_hi=70.0, verbose=True):
    rows = []
    for k, b in enumerate(np.geomspace(b_hi, b_lo, n)):
        try:
            rows.append(solve(b))
        except Exception as exc:                      # skip a bad point
            if verbose:
                print(f"  b={b:.4g}: skipped ({exc})")
            continue
        if verbose:
            d = rows[-1]
            print(f"  [{k + 1}/{n}] R={d['R']:.4f} rc={d['rc']:.4f} "
                  f"theta={np.degrees(d['th']):.1f} Q={d['Q']:.4g}")
    rows.sort(key=lambda d: d["R"])
    return {k: np.array([d[k] for d in rows]) for k in rows[0]}


def load_table():
    if _CACHE.exists():
        return dict(np.load(_CACHE))
    print("Solving Young-Laplace bubble shapes (first run only)...")
    tab = build_table()
    np.savez(_CACHE, **tab)
    return tab


def shape_at(tab, Rn):
    """Interpolated shape for dimensionless radius Rn = R / l_c."""
    lr = np.log(tab["R"])
    x = float(np.clip(np.log(Rn), lr[0], lr[-1]))
    i = int(np.clip(np.searchsorted(lr, x) - 1, 0, len(lr) - 2))
    f = (x - lr[i]) / (lr[i + 1] - lr[i])
    out = {k: (1 - f) * v[i] + f * v[i + 1] for k, v in tab.items()}
    return out


def sim_table(tab, nt=64):
    """Ratios used by the particle model, on a uniform grid in log(R / l_c).

    Columns: r_c / R^2, r_max / R, Q / R^3. They tend to 2/sqrt(3), 1 and 2/3
    for small bubbles, so clamping below the table range stays exact.
    """
    lr = np.log(tab["R"])
    x = np.linspace(lr[0], lr[-1], nt)
    R = np.exp(x)
    cols = [np.interp(x, lr, tab["rc"]) / R**2,
            np.interp(x, lr, tab["rmax"]) / R,
            np.interp(x, lr, tab["Q"]) / R**3]
    return x[0], x[-1], np.stack(cols, axis=1).astype(np.float32)


if __name__ == "__main__":
    t = build_table()
    np.savez(_CACHE, **t)
    R = t["R"]
    print(f"\n{len(R)} shapes, R/l_c from {R[0]:.4f} to {R[-1]:.3f}")
    print("small-bubble limits (expect 1.1547, 1.0, 0.6667):")
    print("  rc/R^2 =", t["rc"][0] / R[0]**2, " rmax/R =", t["rmax"][0] / R[0],
          " Q/R^3 =", t["Q"][0] / R[0]**3)
    print("slope check, Q vs tan(theta)/K1(rc) at smallest:",
          t["Q"][0], np.tan(t["th"][0]) / k1(t["rc"][0]))
    print("monotonic R:", bool(np.all(np.diff(R) > 0)),
          " monotonic rc:", bool(np.all(np.diff(t["rc"]) > 0)))
    for i in range(0, len(R), 6):
        print(f"  R={R[i]:.3f} rc={t['rc'][i]:.3f} theta={np.degrees(t['th'][i]):.1f} "
              f"Rcap={t['Rcap'][i]:.3f} zb={t['zb'][i]:.3f} zc={t['zc'][i]:.4f} "
              f"rmax/R={t['rmax'][i] / R[i]:.3f} Q/R^3={t['Q'][i] / R[i]**3:.3f}")
