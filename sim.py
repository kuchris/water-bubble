"""GPU (Taichi) reduced-order model of bubbles floating on a liquid surface.

Units: mm, mg, s. In these units water has rho = 1, mu = 1, g = 9810 and
surface tension in mN/m is multiplied by 1000 (72.8 mN/m -> 72800).

What is first-principles and what is tuned:
  * Bubble geometry and the capillary "charge" Q come from the Young-Laplace
    table in bubble_shape.py, looked up with the local R / l_c.
  * Bubble-bubble and bubble-wall attraction (the Cheerios effect) use the
    Nicolson superposition: F = 2 pi gamma Q_i Q_j K1(d / l_c) / l_c.
  * Drag is Stokes with the Schiller-Naumann finite-Reynolds correction.
  * Surface tension laws gamma(c) are fits to handbook values.
  * Marangoni spreading (surface velocity = KM * grad gamma), film drainage,
    rupture and coalescence rates are phenomenological and tuned by hand.
"""
import math

import numpy as np
import taichi as ti

import bubble_shape

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=1)

# ---- domain and resolution ------------------------------------------------
L = 100.0                 # tank side, mm
NG = 256                  # concentration grid
DX = L / NG
MAXB = 2048               # bubble slots
VIEW, PANEL = 900, 500    # top view size and side panel width, px
W, H = VIEW + PANEL, VIEW

# ---- physical constants (mm, mg, s) ----------------------------------------
RHO, GRAV, MU_W = 1.0, 9810.0, 1.0
GAMMA_W = 72800.0
PI = math.pi

# ---- phenomenological constants --------------------------------------------
KM = 0.04          # surface velocity per unit grad(gamma), mm^2 s / mg
D0 = 2.0           # background mixing diffusivity, mm^2 / s
TAU_BULK = 15.0    # surface relaxes to the bulk concentration, s
H0, HCRIT = 10.0, 0.1   # initial / critical cap film thickness, micrometres
TAU_W = 0.6        # film drainage time in clean water, s
ASTAB = 40.0       # drainage slow-down for a fully surfactant-covered film
MERGE0 = 0.8      # coalescence rate of touching bubbles in clean water, 1/s
R_MAX = 12.0       # bubbles larger than this break up, mm
POP_T = 0.3        # pop ring animation, s

# ---- materials ---------------------------------------------------------------
# gamma(c) = GAMMA_W + A * ln(1 + B c) / ln(1 + B), c clamped to [0, 1].
# A small B gives a linear law. c = 1 means `cmax` in the unit shown.
#   A, B      surface tension law: A is the total change at c = 1 (negative
#             lowers the tension), B how sharply it happens at low c
#   s_max,    film stability S(c) = s_max c / (c + s_half), between 0 and 1.
#   s_half    Surfactants make the film surfaces rigid, which slows drainage
#             and stops touching bubbles from merging
#   antifoam  fraction by which the additive shortens the drainage time
#   v1, v4    viscosity relative to water = exp(v1 c + v4 c^4)
#   c_inh     concentration above which coalescence is suppressed (the
#             salt transition near 0.1 mol/L)
#   tint      colour the additive gives the liquid on screen
MATERIALS = {
    "Pure water": dict(A=0.0, B=1.0, s_max=0.0, s_half=1.0, antifoam=0.0,
                       v1=0.0, v4=0.0, c_inh=1e9, tint=(0.04, 0.10, 0.18),
                       unit="", cmax=0.0),
    "Soap (SDS)": dict(A=-37800.0, B=20.0, s_max=1.0, s_half=0.15, antifoam=0.0,
                       v1=0.0, v4=0.0, c_inh=1e9, tint=(0.75, 0.25, 0.65),
                       unit="x CMC", cmax=1.0),
    "Ethanol": dict(A=-50800.0, B=100.0, s_max=0.0, s_half=1.0, antifoam=0.7,
                    v1=0.0, v4=0.0, c_inh=1e9, tint=(0.85, 0.60, 0.15),
                    unit="mass frac", cmax=1.0),
    "Salt (NaCl)": dict(A=9800.0, B=0.01, s_max=0.06, s_half=0.05, antifoam=0.0,
                        v1=0.0, v4=0.0, c_inh=0.017, tint=(0.55, 0.80, 0.85),
                        unit="mol/L", cmax=6.0),
    "Glycerol": dict(A=-9400.0, B=0.01, s_max=0.0, s_half=1.0, antifoam=0.0,
                     v1=3.6, v4=3.6, c_inh=1e9, tint=(0.30, 0.75, 0.35),
                     unit="mass frac", cmax=1.0),
}

# The active material lives in one GPU-side struct that every kernel reads, so
# switching liquid or moving a slider needs no recompilation.
Params = ti.types.struct(A=ti.f32, B=ti.f32, s_max=ti.f32, s_half=ti.f32,
                         antifoam=ti.f32, v1=ti.f32, v4=ti.f32, c_inh=ti.f32,
                         c_bulk=ti.f32, wall=ti.f32, tint=ti.math.vec3)
P = Params.field(shape=())

# ---- Young-Laplace lookup table ------------------------------------------------
# Three shape ratios as functions of R / l_c, sampled uniformly in log(R / l_c)
# and interpolated on the GPU. Because l_c depends on the local surface
# tension, the same bubble changes shape as the liquid under it changes.
YL = bubble_shape.load_table()
NT = 64
T_LO, T_HI, _tab = bubble_shape.sim_table(YL, NT)
_tab[:, 2] = np.log(_tab[:, 2])        # Q / R^3 grows exponentially: store its log
tab = ti.Vector.field(3, ti.f32, shape=NT)
tab.from_numpy(_tab)

# ---- fields -----------------------------------------------------------------
# c0 holds the current concentration; c1 is the scratch buffer of the explicit
# diffusion step (read one, write the other, then swap roles).
c0 = ti.field(ti.f32, shape=(NG, NG))      # surface concentration (normalised)
c1 = ti.field(ti.f32, shape=(NG, NG))

# Bubbles are a fixed pool of MAXB slots; `alive` marks the ones in use.
pos = ti.Vector.field(2, ti.f32, shape=MAXB)
vel = ti.Vector.field(2, ti.f32, shape=MAXB)
frc = ti.Vector.field(2, ti.f32, shape=MAXB)
rad = ti.field(ti.f32, shape=MAXB)         # volume-equivalent radius
film = ti.field(ti.f32, shape=MAXB)        # cap film thickness, micrometres
tauf = ti.field(ti.f32, shape=MAXB)        # per-bubble drainage time factor
age = ti.field(ti.f32, shape=MAXB)
popt = ti.field(ti.f32, shape=MAXB)        # remaining pop animation time
alive = ti.field(ti.i32, shape=MAXB)
partner = ti.field(ti.i32, shape=MAXB)
# derived per-bubble quantities, refreshed every substep
b_c = ti.field(ti.f32, shape=MAXB)
b_gam = ti.field(ti.f32, shape=MAXB)
b_lc = ti.field(ti.f32, shape=MAXB)
b_rf = ti.field(ti.f32, shape=MAXB)        # footprint radius (top view)
b_rc = ti.field(ti.f32, shape=MAXB)        # cap (film) radius
b_Q = ti.field(ti.f32, shape=MAXB)         # capillary charge, mm
b_m = ti.field(ti.f32, shape=MAXB)         # effective (added) mass
b_us = ti.Vector.field(2, ti.f32, shape=MAXB)   # Marangoni surface velocity

img = ti.Vector.field(3, ti.f32, shape=(W, H))

# side-view profiles: 0 = current liquid, 1 = pure water reference
NW, NM = bubble_shape.NW, bubble_shape.NM
prof_wall = ti.field(ti.f32, shape=(2, NW))
prof_men = ti.field(ti.f32, shape=(2, NM))
prof_s = ti.field(ti.f32, shape=(2, 6))    # zb, zc, rc, Rcap, theta, meniscus span
SV_X0, SV_X1, SV_Y0, SV_Y1 = VIEW + 20, W - 20, 600, 880
SV_SCALE = (SV_X1 - SV_X0) / 28.0          # px per mm (view is 28 mm wide)
SV_ZMIN = -5.5


# ---- material laws -----------------------------------------------------------
@ti.func
def gamma_of(c):
    cc = ti.math.clamp(c, 0.0, 1.0)
    return GAMMA_W + P[None].A * ti.log(1.0 + P[None].B * cc) / ti.log(1.0 + P[None].B)


@ti.func
def dgamma_dc(c):
    r = 0.0
    if 0.0 <= c < 1.0:
        r = P[None].A * P[None].B / ((1.0 + P[None].B * c) * ti.log(1.0 + P[None].B))
    return r


@ti.func
def mu_rel(c):
    cc = ti.math.clamp(c, 0.0, 1.0)
    return ti.exp(P[None].v1 * cc + P[None].v4 * cc**4)


@ti.func
def stability(c):
    cc = ti.max(c, 0.0)
    return P[None].s_max * cc / (cc + P[None].s_half)


@ti.func
def d_eff(c):
    """Effective diffusivity of the additive along the surface.

    The surface flows toward higher tension with velocity KM * grad(gamma) and
    carries the additive with it. With gamma = gamma(c) the transport flux
    c * KM * gamma'(c) * grad(c) has the form of diffusion with coefficient
    KM * c * |gamma'|, as long as the additive lowers the tension. For one
    that raises it (salt) the same term would un-mix the liquid, so it is
    dropped and only the background mixing D0 remains.
    """
    return D0 + KM * ti.max(c, 0.0) * ti.max(-dgamma_dc(c), 0.0)


@ti.func
def bessel_k1(x):
    """Modified Bessel function K1 (Abramowitz & Stegun 9.8.3, 9.8.7, 9.8.8)."""
    r = 0.0
    if x <= 2.0:
        t = (x / 3.75) ** 2
        i1 = x * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (
            0.02658733 + t * (0.00301532 + t * 0.00032411))))))
        y = (x / 2.0) ** 2
        r = (x * ti.log(x / 2.0) * i1 + 1.0 + y * (0.15443144 + y * (-0.67278579 + y * (
            -0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * -0.00004686)))))) / x
    else:
        y = 2.0 / x
        r = ti.exp(-x) / ti.sqrt(x) * (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (
            0.01504268 + y * (-0.00780353 + y * (0.00325614 + y * -0.00068245))))))
    return r


@ti.kernel
def k1_probe(x: ti.f32) -> ti.f32:
    return bessel_k1(x)


# ---- concentration field -----------------------------------------------------
@ti.func
def sample_c(p):
    """Bilinear concentration and its gradient at position p (mm)."""
    g = p / DX - 0.5
    i = ti.math.clamp(ti.cast(ti.floor(g.x), ti.i32), 0, NG - 2)
    j = ti.math.clamp(ti.cast(ti.floor(g.y), ti.i32), 0, NG - 2)
    fx = ti.math.clamp(g.x - i, 0.0, 1.0)
    fy = ti.math.clamp(g.y - j, 0.0, 1.0)
    a, b, cc, d = c0[i, j], c0[i + 1, j], c0[i, j + 1], c0[i + 1, j + 1]
    val = (a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy
    grad = ti.Vector([((b - a) * (1 - fy) + (d - cc) * fy) / DX,
                      ((cc - a) * (1 - fx) + (d - b) * fx) / DX])
    return val, grad


@ti.func
def _diffuse_step(a: ti.template(), b: ti.template(), dt: ti.f32):
    """One explicit finite-volume step, reading field a and writing field b."""
    for i, j in a:
        cc = a[i, j]
        dc = d_eff(cc)
        acc = 0.0
        for off in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
            # Clamping the index makes a boundary cell its own neighbour:
            # zero flux through the tank walls.
            cn = a[ti.math.clamp(i + off[0], 0, NG - 1), ti.math.clamp(j + off[1], 0, NG - 1)]
            # Flux through the shared face uses the mean diffusivity of the two
            # cells, so what leaves one cell enters the other exactly.
            acc += 0.5 * (dc + d_eff(cn)) * (cn - cc)
        # Last term: the surface slowly exchanges additive with the liquid below.
        b[i, j] = cc + dt * (acc / (DX * DX) + (P[None].c_bulk - cc) / TAU_BULK)


@ti.kernel
def diffuse8(dt: ti.f32):
    """Eight explicit steps of d c/dt = div(D_eff grad c) + bulk exchange.

    Batching eight steps into one kernel cuts the launch overhead, which
    matters because the stability limit forces hundreds of steps per frame.
    The static loop is unrolled at compile time, ending with the result in c0.
    """
    for _ in ti.static(range(4)):
        _diffuse_step(c0, c1, dt)
        _diffuse_step(c1, c0, dt)


@ti.kernel
def fill_c(v: ti.f32):
    for i, j in c0:
        c0[i, j] = v


@ti.kernel
def shift_c(dv: ti.f32):
    for i, j in c0:
        c0[i, j] = ti.max(c0[i, j] + dv, 0.0)


@ti.kernel
def add_drop(x: ti.f32, y: ti.f32, amount: ti.f32, sigma: ti.f32):
    """Add a Gaussian blob of additive. Values above 1 act as a reservoir:
    the tension law is flat there, so the excess only feeds the spreading edge."""
    for i, j in c0:
        d2 = ((i + 0.5) * DX - x) ** 2 + ((j + 0.5) * DX - y) ** 2
        c0[i, j] = ti.min(c0[i, j] + amount * ti.exp(-d2 / (2 * sigma * sigma)), 1.5)


def max_diffusivity():
    """Upper bound of D_eff for the current material (for the stable step)."""
    p = P[None]
    return D0 + KM * abs(p.A) * p.B / ((1 + p.B) * math.log(1 + p.B)) * (1.0 if p.A < 0 else 0.0)


# ---- bubbles -------------------------------------------------------------------
@ti.func
def table(rn):
    """Shape ratios (r_c / R^2, r_max / R, log Q / R^3) for rn = R / l_c."""
    t = (ti.log(rn) - T_LO) / (T_HI - T_LO) * (NT - 1)
    t = ti.math.clamp(t, 0.0, NT - 1.001)
    i = ti.cast(t, ti.i32)
    return tab[i] * (1 - (t - i)) + tab[i + 1] * (t - i)


@ti.kernel
def derive():
    """Refresh what each bubble needs from the liquid directly beneath it."""
    for i in range(MAXB):
        if alive[i]:
            c, gc = sample_c(pos[i])
            g = gamma_of(c)
            lc = ti.sqrt(g / (RHO * GRAV))       # local capillary length
            rn = rad[i] / lc
            t = table(rn)
            b_c[i], b_gam[i], b_lc[i] = c, g, lc
            # The table is dimensionless (lengths in l_c); convert back to mm.
            b_rc[i] = t[0] * rn * rn * lc
            b_rf[i] = ti.max(t[1] * rad[i], b_rc[i])
            b_Q[i] = ti.exp(t[2]) * rn**3 * lc
            # A body accelerating in liquid drags liquid along; for a sphere
            # that added mass is half the displaced mass. The gas weighs nothing.
            b_m[i] = 0.5 * RHO * 4.0 / 3.0 * PI * rad[i] ** 3
            # Marangoni surface flow: grad(gamma) = gamma'(c) grad(c).
            b_us[i] = KM * dgamma_dc(c) * gc


@ti.kernel
def forces(dt: ti.f32):
    """Capillary and contact forces on every bubble, all pairs (O(N^2)).

    The outer loop runs in parallel on the GPU, one thread per bubble; each
    thread only writes its own frc[i], so there are no write conflicts.
    """
    for i in range(MAXB):
        if alive[i]:
            f = ti.Vector([0.0, 0.0])
            p = pos[i]
            for j in range(MAXB):
                if j != i and alive[j]:
                    d = pos[j] - p
                    dist = d.norm() + 1e-6
                    # Pair properties are symmetric averages so that the force
                    # on i from j is equal and opposite to that on j from i.
                    lij = 0.5 * (b_lc[i] + b_lc[j])
                    rsum = b_rf[i] + b_rf[j]
                    if dist < rsum + 7.0 * lij:       # K1 is negligible beyond
                        n = d / dist
                        gij = 0.5 * (b_gam[i] + b_gam[j])
                        # Each bubble sits in the tilted meniscus of the other
                        # and climbs it. K1 diverges at zero distance, so the
                        # argument is not allowed below a fraction of contact.
                        x = ti.max(dist, 0.6 * rsum) / lij
                        # far-field formula overshoots near contact for R > l_c:
                        # a capillary force cannot exceed ~ gamma * size
                        fa = 2 * PI * gij * b_Q[i] * b_Q[j] * bessel_k1(x) / lij
                        f += n * ti.min(fa, gij * ti.min(b_rf[i], b_rf[j]))
                        ov = rsum - dist
                        if ov > 0:
                            # Contact: a spring as stiff as the explicit time
                            # step allows (k dt^2 / m = 0.2), plus damping of
                            # the approach velocity so pairs do not bounce.
                            mred = b_m[i] * b_m[j] / (b_m[i] + b_m[j])
                            k = 0.2 * mred / (dt * dt)
                            vrel = (vel[j] - vel[i]).dot(n)
                            f += n * (-k * ov + ti.sqrt(k * mred) * vrel)
            # walls: wetted wall meniscus z = l_c exp(-x / l_c) attracts bubbles
            kw = 0.2 * b_m[i] / (dt * dt)
            for a in ti.static(range(2)):
                for s in ti.static((-1.0, 1.0)):
                    xw = p[a] if s < 0 else L - p[a]
                    e = ti.Vector([0.0, 0.0])
                    e[a] = s
                    xe = ti.max(xw, b_rf[i])
                    fw = 2 * PI * b_gam[i] * b_Q[i] * ti.exp(-xe / b_lc[i])
                    f += e * P[None].wall * ti.min(fw, b_gam[i] * b_rf[i])
                    if xw < b_rf[i]:
                        f -= e * kw * (b_rf[i] - xw)
            frc[i] = f


@ti.kernel
def integrate(dt: ti.f32):
    """Advance velocities and positions: m dv/dt = F - zeta (v - u_surface)."""
    for i in range(MAXB):
        if alive[i]:
            mu = MU_W * mu_rel(b_c[i])
            # Drag acts on the velocity relative to the moving surface, which
            # is how Marangoni flow carries bubbles along.
            rel = vel[i] - b_us[i]
            re = RHO * rel.norm() * 2.0 * rad[i] / mu
            # Half the Stokes drag of a sphere (the bubble is only partly
            # surrounded by liquid) times the Schiller-Naumann correction,
            # which accounts for inertia of the flow at Reynolds numbers > 1.
            zeta = 3.0 * PI * mu * rad[i] * (1.0 + 0.15 * re**0.687)
            # The drag term is treated implicitly (it appears on both sides and
            # is solved for the new velocity). Small bubbles have a drag time
            # far shorter than the step; an explicit update would blow up.
            a = dt * zeta / b_m[i]
            vel[i] = (vel[i] + dt * frc[i] / b_m[i] + a * b_us[i]) / (1.0 + a)
            q = pos[i] + dt * vel[i]
            lo = 0.3 * b_rf[i]
            pos[i] = ti.Vector([ti.math.clamp(q.x, lo, L - lo), ti.math.clamp(q.y, lo, L - lo)])


@ti.kernel
def separate():
    """Bubbles cannot interpenetrate: push overlapping pairs apart (one Jacobi pass).

    The contact spring alone is too soft inside a large raft, where many
    neighbours pull inward at once. Each bubble is moved out of its overlaps,
    the lighter one of a pair moving more. All corrections are computed from
    the old positions first (stored in frc, free at this point) and applied
    afterwards, so the result does not depend on thread order.
    """
    for i in range(MAXB):
        if alive[i]:
            d = ti.Vector([0.0, 0.0])
            for j in range(MAXB):
                if j != i and alive[j]:
                    r = pos[i] - pos[j]
                    dist = r.norm() + 1e-6
                    ov = b_rf[i] + b_rf[j] - dist
                    if ov > 0:
                        d += r / dist * (0.4 * ov * b_m[j] / (b_m[i] + b_m[j]))
            frc[i] = d
    for i in range(MAXB):
        if alive[i]:
            pos[i] += frc[i]


@ti.kernel
def drain(dt: ti.f32) -> ti.i32:
    """Thin the cap films and burst the ones that fail. Returns the burst count."""
    n = 0
    for i in range(MAXB):
        if alive[i]:
            c = b_c[i]
            # Drainage time of the cap film: slower in a viscous liquid, much
            # slower when surfactant stiffens the film surfaces, faster with an
            # antifoam. tauf is the bubble's own random factor.
            af = 1.0 - P[None].antifoam * ti.min(ti.max(c, 0.0) / 0.3, 1.0)
            taud = TAU_W * tauf[i] * mu_rel(c) * (1.0 + ASTAB * stability(c)) * af
            film[i] -= dt * film[i] / taud            # exponential thinning
            age[i] += dt
            # Rupture is random: the chance per second rises steeply as the
            # film nears the critical thickness, and is certain below it.
            hazard = (3.0 * HCRIT / film[i]) ** 2 / taud
            if film[i] < HCRIT or rad[i] > R_MAX or ti.random() < 1.0 - ti.exp(-hazard * dt):
                alive[i] = 0
                popt[i] = POP_T
                n += 1
        elif popt[i] > 0:
            popt[i] -= dt
    return n


@ti.func
def hash01(a, b, f):
    """Pseudo-random number in [0, 1) that is a pure function of its inputs.

    Both bubbles of a pair must draw the same number to agree on whether they
    merge; ti.random() would give each GPU thread a different one.
    """
    x = ti.cast(a, ti.u32) * ti.u32(73856093) ^ ti.cast(b, ti.u32) * ti.u32(19349663) \
        ^ ti.cast(f, ti.u32) * ti.u32(83492791)
    x ^= x >> 16
    x *= ti.u32(0x7FEB352D)
    x ^= x >> 15
    x *= ti.u32(0x846CA68B)
    x ^= x >> 16
    return ti.cast(x >> 8, ti.f32) / 16777216.0


@ti.kernel
def pick_partners(dt: ti.f32, frame: ti.i32):
    """Each bubble picks at most one touching neighbour to coalesce with.

    Merging changes two bubbles at once, which is unsafe to do directly in a
    parallel loop. So it is split: here every bubble only records a choice
    (the touching neighbour with the lowest shared random draw that passes the
    merge probability), and merge() acts on pairs that chose each other.
    """
    for i in range(MAXB):
        best, score = -1, 2.0
        if alive[i]:
            for j in range(MAXB):
                if j != i and alive[j]:
                    dist = (pos[j] - pos[i]).norm()
                    if dist < 1.02 * (b_rf[i] + b_rf[j]):
                        c = 0.5 * (b_c[i] + b_c[j])
                        # The liquid film between the two must drain first:
                        # slower when viscous, blocked by surfactant or salt.
                        rate = MERGE0 / mu_rel(c) * (1.0 - stability(c)) ** 6 \
                            / (1.0 + (ti.max(c, 0.0) / P[None].c_inh) ** 2)
                        u = hash01(ti.min(i, j), ti.max(i, j), frame)
                        if u < 1.0 - ti.exp(-rate * dt) and u < score:
                            best, score = j, u
        partner[i] = best


@ti.kernel
def merge() -> ti.i32:
    """Fuse mutually chosen pairs into the lower slot. Returns the merge count."""
    n = 0
    for i in range(MAXB):
        j = partner[i]
        # Mutual choice means no third bubble touches this pair's data, and
        # j > i makes exactly one of the two threads do the work.
        if j > i and partner[j] == i:
            # Volume is conserved; position and velocity are volume-weighted.
            vi, vj = rad[i] ** 3, rad[j] ** 3
            wi = vi / (vi + vj)
            pos[i] = pos[i] * wi + pos[j] * (1 - wi)
            vel[i] = vel[i] * wi + vel[j] * (1 - wi)
            film[i] = film[i] * wi + film[j] * (1 - wi)
            tauf[i] = ti.max(tauf[i], tauf[j])
            rad[i] = (vi + vj) ** (1.0 / 3.0)
            alive[j] = 0
            popt[j] = 0.0
            n += 1
    return n


@ti.kernel
def spawn(slots: ti.types.ndarray(dtype=ti.i32, ndim=1),
          data: ti.types.ndarray(dtype=ti.f32, ndim=2), n: ti.i32):
    for k in range(n):
        i = slots[k]
        pos[i] = ti.Vector([data[k, 0], data[k, 1]])
        vel[i] = ti.Vector([0.0, 0.0])
        rad[i] = data[k, 2]
        tauf[i] = data[k, 3]
        film[i], age[i], popt[i], alive[i] = H0, 0.0, 0.0, 1


@ti.kernel
def clear_bubbles():
    for i in range(MAXB):
        alive[i] = 0
        popt[i] = 0.0


@ti.kernel
def count_alive() -> ti.i32:
    n = 0
    for i in range(MAXB):
        n += alive[i]
    return n


# ---- rendering -----------------------------------------------------------------
@ti.func
def gamma_colour(g):
    """Dark purple (low tension) -> teal -> yellow (high tension)."""
    t = ti.math.clamp((g - 20000.0) / 65000.0, 0.0, 1.0)
    lo = ti.Vector([0.27, 0.05, 0.40])
    mid = ti.Vector([0.13, 0.57, 0.55])
    hi = ti.Vector([0.95, 0.88, 0.20])
    col = lo * (1 - 2 * t) + mid * (2 * t)
    if t > 0.5:
        col = mid * (2 - 2 * t) + hi * (2 * t - 1)
    return col


@ti.func
def film_colour(h):
    """Thin-film interference colour for a water film of thickness h (um).

    Light reflected from the two faces of the film interferes; the reflected
    intensity at wavelength lam goes as sin^2(2 pi n h / lam) with n = 1.33.
    Evaluated for red, green and blue. Thick films mix many orders and look
    silvery, very thin ones reflect almost nothing ("black film").
    """
    lam = ti.Vector([0.65, 0.53, 0.45])
    col = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        col[k] = ti.sin(2 * PI * 1.33 * h / lam[k]) ** 2
    w = ti.math.smoothstep(1.0, 3.0, h)
    return col * (1 - w) + ti.Vector([0.80, 0.85, 0.90]) * w


@ti.func
def region(k, r, z):
    """Side view: 0 = air, 1 = liquid, 2 = gas inside the bubble.

    Classifies the point (r, z) of the cross-section of profile k. Below the
    junction ring the bubble is bounded by the tabulated wall, above it by the
    spherical cap; outside the ring the liquid reaches up to the meniscus.
    """
    zb, zc, rc, rcap, th = prof_s[k, 0], prof_s[k, 1], prof_s[k, 2], prof_s[k, 3], prof_s[k, 4]
    res = 0
    gas = False
    if zb <= z <= zc:
        t = ti.min((z - zb) / (zc - zb) * (NW - 1), NW - 1.001)
        i = ti.cast(t, ti.i32)
        gas = r < prof_wall[k, i] * (1 - (t - i)) + prof_wall[k, i + 1] * (t - i)
    elif z > zc and r < rc:
        gas = r * r + (z - (zc - rcap * ti.cos(th))) ** 2 < rcap * rcap
    if gas:
        res = 2
    else:
        surf = zc
        if r >= rc:
            t = (r - rc) / prof_s[k, 5] * (NM - 1)
            surf = 0.0
            if t < NM - 1.001:
                i = ti.cast(t, ti.i32)
                surf = prof_men[k, i] * (1 - (t - i)) + prof_men[k, i + 1] * (t - i)
        if z < surf:
            res = 1
    return res


@ti.kernel
def render(show_gamma: ti.i32):
    """Draw the whole frame into `img` on the GPU."""
    # top view: liquid surface, tinted by concentration (or coloured by
    # tension), with spreading fronts brightened where the gradient is steep
    for x, y in ti.ndrange(VIEW, VIEW):
        c, gc = sample_c(ti.Vector([x + 0.5, y + 0.5]) * (L / VIEW))
        col = ti.Vector([0.04, 0.10, 0.18])
        if show_gamma:
            col = gamma_colour(gamma_of(c)) * 0.8
        else:
            col = col + (P[None].tint - col) * ti.min(ti.max(c, 0.0), 1.2) * 0.6
        front = ti.min(ti.abs(dgamma_dc(c)) * gc.norm() / 6000.0, 1.0)
        img[x, y] = col + 0.25 * front

    # top view: bubbles and pop rings. Each bubble paints only the pixels of
    # its own bounding box; bubbles barely overlap, so parallel threads rarely
    # touch the same pixel and the result is stable.
    s = VIEW / L
    for i in range(MAXB):
        if alive[i] or popt[i] > 0:
            cx, cy = pos[i].x * s, pos[i].y * s
            grow = 1.0
            if not alive[i]:
                grow = 1.0 + 0.8 * (1.0 - popt[i] / POP_T)
            rp = b_rf[i] * s * grow
            x0, x1 = ti.max(int(cx - rp - 2), 0), ti.min(int(cx + rp + 2), VIEW - 1)
            y0, y1 = ti.max(int(cy - rp - 2), 0), ti.min(int(cy + rp + 2), VIEW - 1)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    dx, dy = (x + 0.5 - cx) / rp, (y + 0.5 - cy) / rp
                    d = ti.sqrt(dx * dx + dy * dy)
                    base = img[x, y]
                    if alive[i]:
                        aa = ti.math.clamp((1.0 - d) * rp + 0.5, 0.0, 1.0)
                        if aa > 0:
                            # body with a bright rim, the film cap coloured by
                            # its thickness (thinner at the top, hence rings),
                            # and a highlight
                            rim = ti.math.smoothstep(0.7, 1.0, d)
                            col = base * 1.15 + 0.04 + ti.Vector([0.55, 0.68, 0.80]) * rim * 0.55
                            dc = d * b_rf[i] / b_rc[i]
                            if dc < 1.0:
                                fc = film_colour(film[i] * (0.7 + 0.6 * dc * dc))
                                edge = ti.math.clamp((1.0 - dc) * b_rc[i] * s, 0.0, 1.0)
                                col = col + (fc - col) * 0.5 * edge
                            hl = ti.exp(-((dx + 0.35) ** 2 + (dy - 0.38) ** 2) / 0.025)
                            col += 0.7 * hl
                            img[x, y] = base + (col - base) * aa
                    else:
                        ring = ti.math.clamp(1.2 - ti.abs(1.0 - d) * rp, 0.0, 1.0)
                        img[x, y] = base + ring * 0.6 * popt[i] / POP_T

    # side panel: fill each pixel by region, and draw an outline wherever the
    # region changes between neighbouring pixels
    liquid = ti.Vector([0.05, 0.16, 0.30])
    liquid = liquid + (P[None].tint - liquid) * ti.min(P[None].c_bulk, 1.0) * 0.45
    for x, y in ti.ndrange((VIEW, W), H):
        col = ti.Vector([0.07, 0.08, 0.10])
        if SV_X0 <= x < SV_X1 and SV_Y0 <= y < SV_Y1:
            r = ti.abs((x + 0.5 - 0.5 * (SV_X0 + SV_X1)) / SV_SCALE)
            z = (y + 0.5 - SV_Y0) / SV_SCALE + SV_ZMIN
            dp = 1.0 / SV_SCALE
            a = region(0, r, z)
            col = ti.Vector([0.10, 0.11, 0.14])
            if a == 1:
                col = liquid
            elif a == 2:
                col = ti.Vector([0.16, 0.18, 0.22])
            g0 = region(1, r, z)
            if (g0 != region(1, r + dp, z) or g0 != region(1, r, z + dp)) and (x + y) % 6 < 3:
                col = ti.Vector([0.55, 0.57, 0.60])           # pure-water reference
            if a != region(0, r + dp, z) or a != region(0, r, z + dp) \
                    or a != region(0, r - dp, z) or a != region(0, r, z - dp):
                col = ti.Vector([0.95, 0.97, 1.0])
            # 5 mm scale bar
            if SV_Y0 + 14 <= y < SV_Y0 + 17 and SV_X0 + 14 <= x < SV_X0 + 14 + 5 * SV_SCALE:
                col = ti.Vector([0.9, 0.9, 0.9])
        img[x, y] = col


# ---- host-side helpers ---------------------------------------------------------
def set_material(name, c_bulk, wall=True):
    m = MATERIALS[name]
    P[None] = Params(A=m["A"], B=m["B"], s_max=m["s_max"], s_half=m["s_half"],
                     antifoam=m["antifoam"], v1=m["v1"], v4=m["v4"], c_inh=m["c_inh"],
                     c_bulk=c_bulk, wall=1.0 if wall else 0.0, tint=ti.Vector(m["tint"]))


def gamma_host(name, c):
    """Surface tension in mN/m for material `name` at normalised concentration c."""
    m = MATERIALS[name]
    cc = min(max(c, 0.0), 1.0)
    return (GAMMA_W + m["A"] * math.log(1 + m["B"] * cc) / math.log(1 + m["B"])) / 1000.0


def set_profile(slot, radius_mm, gamma_mn):
    """Upload the side-view shape of a bubble of given radius and surface tension."""
    lc = math.sqrt(gamma_mn * 1000.0 / (RHO * GRAV))
    sh = bubble_shape.shape_at(YL, radius_mm / lc)
    scale = radius_mm / sh["R"]       # exact within the table, self-similar outside
    w = prof_wall.to_numpy()
    m = prof_men.to_numpy()
    sc = prof_s.to_numpy()
    w[slot] = sh["wall_r"] * scale
    m[slot] = sh["men_z"] * scale
    sc[slot] = [sh["zb"] * scale, sh["zc"] * scale, sh["rc"] * scale,
                sh["Rcap"] * scale, sh["th"], bubble_shape.MEN_SPAN * scale]
    prof_wall.from_numpy(w)
    prof_men.from_numpy(m)
    prof_s.from_numpy(sc)
    return dict(lc=lc, bond=(radius_mm / lc) ** 2, rc=sh["rc"] * scale,
                theta=math.degrees(sh["th"]), height=(sh["zc"] + sh["Rcap"] * (1 - math.cos(sh["th"]))) * scale,
                depth=-sh["zb"] * scale)


_rng = np.random.default_rng(0)


def spawn_bubbles(xy, radii):
    """Add bubbles at positions xy (n, 2) mm with the given radii. Returns count added."""
    free = np.flatnonzero(alive.to_numpy() == 0).astype(np.int32)
    n = min(len(free), len(radii))
    if n == 0:
        return 0
    data = np.zeros((n, 4), dtype=np.float32)
    data[:, :2] = np.clip(xy[:n], 1.0, L - 1.0)
    data[:, 2] = radii[:n]
    data[:, 3] = np.exp(0.5 * _rng.standard_normal(n))
    spawn(free[:n].copy(), data, n)
    return n


def random_radii(n, mean):
    return np.clip(mean * np.exp(0.35 * _rng.standard_normal(n)), 0.15, 8.0).astype(np.float32)


def random_xy(n, centre=None, spread=None):
    if centre is None:
        return _rng.uniform(4.0, L - 4.0, size=(n, 2)).astype(np.float32)
    return (np.asarray(centre) + spread * _rng.standard_normal((n, 2))).astype(np.float32)


_frame = 0


def step(dt, substeps=8):
    """Advance the whole model by dt seconds. Returns (bursts, merges)."""
    global _frame
    _frame += 1
    # 1. Additive transport. Explicit diffusion is stable only for steps below
    #    dx^2 / (4 D); 0.2 leaves a margin. Fast Marangoni spreading makes D
    #    large, so one frame takes many small steps.
    dt_d = 0.2 * DX * DX / max_diffusivity()
    calls = max(1, math.ceil(dt / dt_d / 8))
    for _ in range(calls):
        diffuse8(dt / (calls * 8))
    # 2. Bubble motion, in smaller steps than the frame so contacts stay stiff.
    h = dt / substeps
    for _ in range(substeps):
        derive()
        forces(h)
        integrate(h)
        separate()
    # 3. Slow processes once per frame: film drainage, bursting, coalescence.
    derive()
    bursts = drain(dt)
    pick_partners(dt, _frame)
    merges = merge()
    return bursts, merges
