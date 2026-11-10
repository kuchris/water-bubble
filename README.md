# Water bubble

An interactive GPU simulation of bubbles floating on a liquid surface, showing
how dissolved substances change their behaviour through surface tension, film
stability and viscosity.

![Soap solution: bubbles gather into rafts and stick to the walls](docs/screenshot.png)

The point it makes: **low surface tension does not mean long-lived bubbles.**
Soap and ethanol both lower the surface tension of water, but soap bubbles last
for minutes while bubbles on an ethanol solution burst faster than on clean
water. What keeps a bubble alive is the stability of its thin film, not the
tension value.

## Run

Needs [uv](https://docs.astral.sh/uv/) and a GPU (CUDA or Vulkan; Taichi falls
back to the CPU if neither is available).

```bash
uv run main.py
```

The first start solves the bubble shape table (about 15 s) and caches it in
`yl_table.npz`.

| Input | Action |
|---|---|
| Left mouse button | add bubbles at the cursor |
| Right mouse button | add a drop of the current additive |
| `Space` | pause |
| `C` | clear bubbles |
| `R` | reset the liquid to the bulk concentration |
| `Esc` | quit |

The panel on the right selects the liquid (pure water, soap, ethanol, salt,
glycerol), its concentration, the bubble emission rate and size, and the time
scale. The side view above it shows the equilibrium cross-section of a single
bubble in the current liquid, with the same bubble in pure water dashed.

## What you can see

- **Rafts**: bubbles attract each other and the walls through the menisci they
  create (the "Cheerios effect"), over a range set by the capillary length
  `sqrt(gamma / (rho g))`.
- **Shape**: small bubbles sit almost fully submerged as spheres, large ones
  form a dome. The shape depends only on `R / l_c`, so lowering the surface
  tension makes a bubble of the same volume flatter.
- **Marangoni flow**: a drop of soap or ethanol lowers the tension locally and
  the surface flows outward, sweeping bubbles away.
- **Lifetime**: the colour rings on each cap are thin-film interference; they
  move faster as the film drains, and the bubble bursts when it gets too thin.
- **Coalescence**: touching bubbles merge quickly in clean water, rarely in
  soap or salt solutions.

## Model

A reduced-order model, not a CFD solver. Each bubble is a particle on a 2D
surface; the additive is a concentration field on a grid.

Grounded in theory or handbook data:

- Bubble geometry from the axisymmetric Young-Laplace equations for the
  submerged wall, the film cap and the outer meniscus (`bubble_shape.py`). The
  solver reproduces the small-bubble limits `r_cap = (2/sqrt(3)) R^2 / l_c` and
  capillary charge `Q = (2/3) R^3 / l_c^2` to four digits.
- Capillary attraction by Nicolson superposition:
  `F = 2 pi gamma Q_i Q_j K1(d / l_c) / l_c`.
- Stokes drag with the Schiller-Naumann finite-Reynolds correction.
- Surface tension against concentration for each additive.

Phenomenological, tuned by hand:

- Marangoni spreading speed (surface velocity proportional to the tension
  gradient).
- Film drainage time, rupture hazard and coalescence rate, and how each
  additive scales them.
- A cap on the capillary force near contact, where the far-field formula
  overestimates it for bubbles larger than the capillary length.

These give qualitatively correct behaviour. Lifetimes and spreading speeds are
not calibrated against experiments.

## Files

| File | Contents |
|---|---|
| `main.py` | window, controls, scripted test |
| `sim.py` | Taichi fields and kernels: concentration field, bubble dynamics, rendering |
| `bubble_shape.py` | Young-Laplace shape solver and lookup table |

`uv run main.py --test out` runs every liquid for 10 s of model time without a
window, prints bubble counts and writes frames to `out/`.
`uv run python bubble_shape.py` rebuilds the shape table and prints the checks
against the analytic limits.
