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
    """Integrate the wall from the apex (curvature jump b) until it turns over.

    The curve is parametrised by arc length s, starting at the lowest point of
    the bubble. State y = [r, zeta, phi, V]:
      r    distance from the axis
      zeta height above the apex
      phi  angle of the tangent from horizontal (0 at the apex, pi/2 at the
           equator, pi where the wall has turned back to horizontal)
      V    volume enclosed below the current height
    The pressure jump across the wall at height zeta is b + zeta: b at the
    apex, growing upward because the liquid pressure falls hydrostatically.
    Measuring height from the apex makes the curve depend on b alone, so it is
    integrated once and reused for every trial depth in solve().
    """
    def f(s, y):
        r, ze, ph, _ = y
        sp = np.sin(ph)
        # Young-Laplace: total curvature = pressure jump. The total curvature
        # is d(phi)/ds (in the drawing plane) + sin(phi)/r (around the axis).
        return [np.cos(ph), sp, b + ze - sp / r, np.pi * r * r * sp]

    def top(s, y):                 # stop when the wall is horizontal again
        return y[2] - np.pi
    top.terminal = True

    def equator(s, y):             # recorded, the junction lies above it
        return y[2] - 0.5 * np.pi

    # sin(phi)/r is 0/0 on the axis, so start a tiny step away from the apex
    # using the series of a sphere of radius 2/b.
    e = 1e-4 * min(1.0, 2.0 / b)
    y0 = [e, b * e * e / 4, b * e / 2, 0.0]
    sol = solve_ivp(f, (e, 400.0), y0, events=(top, equator), dense_output=True,
                    rtol=1e-10, atol=1e-13, max_step=0.05 * max(1.0, 2.0 / b))
    if not sol.t_events[0].size or not sol.t_events[1].size:
        raise RuntimeError(f"wall integration failed for b={b}")
    return sol.sol, sol.t_events[1][0], sol.t_events[0][0]


def _meniscus(rc, zc, psi, dense=False):
    """Shoot the meniscus outward. Returns +1 if it ends too high, -1 too low.

    Starts at the junction ring (radius rc, height zc) descending at angle psi
    below horizontal. The true meniscus decays to z = 0 at infinity, but that
    solution is unstable when integrated outward: any error grows like e^r.
    The instability is used as the test. A start that is too high flattens out
    (psi reaches 0) while still above the level and then curls upward; one that
    is too low crosses z = 0 while still descending. Whichever happens first
    tells solve() which way to move.
    """
    if zc <= 0:
        return -1, None
    if psi <= 1e-12:
        return +1, None

    def f(s, y):
        r, z, p = y
        # Liquid at height z is at pressure -z relative to the air, and the
        # surface curvature must balance it: d(psi)/ds + sin(psi)/r = -z.
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
    """Point on the wall where a cap of radius 4/P joins tangentially.

    The cap is a soap film with two faces, so the gas pressure P is held by a
    sphere of radius 4/P (a single interface would give 2/P). A sphere centred
    on the axis has r = R_cap sin(angle), so the junction is where the wall
    satisfies r = (4/P) sin(phi). Buoyancy makes the wall flatten out at a
    finite radius instead of closing on the axis, which is why such a point
    exists between the equator and the top.
    """
    def g(s):
        y = wall(s)
        return y[0] - (4.0 / P) * np.sin(y[2])
    if g(s_eq) >= 0:          # cap would exceed a hemisphere
        return None
    return brentq(g, s_eq, s_end, xtol=1e-14)


def solve(b):
    """Full shape for apex curvature jump b. Returns a dict (units of l_c).

    b selects one member of the one-parameter family of bubbles (large b =
    small bubble). The remaining unknown is how deep the bubble sits: zb, the
    height of its apex relative to the far liquid level. It is found by
    bisection so that the meniscus leaving the junction decays to that level.

    Returned keys: R volume-equivalent radius, P gas pressure, zb / zc heights
    of the apex and the junction ring, rc junction (cap) radius, th cap
    half-angle, Rcap cap radius of curvature, rmax widest radius, Q capillary
    charge, wall_r and men_z the sampled profiles.
    """
    wall, s_eq, s_end = _wall(b)
    ze_top = wall(s_end)[1]

    def trial(zb):
        # The apex jump is b = P + zb (gas pressure minus liquid pressure at
        # depth zb), so the depth fixes the gas pressure and with it the cap.
        P = b - zb
        sc = _junction(wall, s_eq, s_end, P)
        if sc is None:
            return -1, None
        r, ze, ph, V = wall(sc)
        # The meniscus leaves tangent to the wall: its descent angle is pi - phi.
        return _meniscus(r, zb + ze, np.pi - ph)[0], sc

    # Bracket: lowest puts the whole wall below the level (meniscus too low),
    # highest is the limit P -> 0 (flat cap, meniscus too high).
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
    # gas volume = part enclosed by the wall + the spherical cap above the ring
    hcap = Rcap * (1 - np.cos(th))
    V = Vw + np.pi * hcap * hcap * (3 * Rcap - hcap) / 3
    R = (3 * V / (4 * np.pi)) ** (1 / 3)

    # wall radius on a uniform z grid (height is monotonic along the wall, so
    # a single r per height describes it; used for drawing the side view)
    s = np.linspace(0.0, sc, 4000)[1:]
    ys = wall(np.maximum(s, 1e-9 * sc))
    ze_s = np.concatenate([[0.0], ys[1]])
    r_s = np.concatenate([[0.0], ys[0]])
    wall_r = np.interp(np.linspace(0.0, ze_c, NW), ze_s, r_s)
    rmax = max(r_s.max(), rc)

    # Meniscus height on a uniform r grid. Far from the bubble the slope is
    # small and the equation becomes linear with solution z = Q K0(r). Q is
    # read off the numerical solution at the matching radius; it measures how
    # strongly the bubble deforms the surface and sets the attraction between
    # bubbles. Past the matching radius the K0 tail replaces the shot solution,
    # which would eventually diverge.
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
    """Solve n shapes spanning R / l_c from about 0.03 to 4.9, sorted by R."""
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
    """Shape table from the cache file, solving and saving it on first use."""
    if _CACHE.exists():
        return dict(np.load(_CACHE))
    print("Solving Young-Laplace bubble shapes (first run only)...")
    tab = build_table()
    np.savez(_CACHE, **tab)
    return tab


def shape_at(tab, Rn):
    """Interpolated shape for dimensionless radius Rn = R / l_c.

    Linear in log(R) between the two neighbouring table entries, applied to
    every scalar and to the sampled profiles alike. Outside the table range the
    nearest entry is returned; the caller rescales it to the requested size.
    """
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
