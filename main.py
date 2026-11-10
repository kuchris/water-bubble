"""Interactive desktop window for the surface-bubble model.

    uv run main.py              interactive window
    uv run main.py --test DIR   scripted run without a window, writes PNGs to DIR
    uv run main.py --shot PNG   short scripted run, saves one screenshot with the UI
"""
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

import sim

NAMES = list(sim.MATERIALS)


class State:
    """Everything the user can change. `dirty` marks a change that still has
    to be pushed to the GPU (done once per frame by apply())."""
    material = "Pure water"
    c_bulk = 0.0
    wall = True
    show_gamma = False
    paused = False
    emit_rate = 15.0      # bubbles per second
    mean_r = 1.2          # mm
    probe_r = 2.0         # mm, bubble shown in the side view
    drop = 0.6            # concentration added per second while RMB is held
    speed = 1.0
    info = {}
    dirty = True


def apply(st):
    """Push UI state to the GPU and refresh the side-view profiles."""
    sim.set_material(st.material, st.c_bulk, st.wall)
    st.info = sim.set_profile(0, st.probe_r, sim.gamma_host(st.material, st.c_bulk))
    sim.set_profile(1, st.probe_r, sim.GAMMA_W / 1000.0)
    st.dirty = False


def select(st, name):
    """Switch to another additive. The model tracks one additive at a time, so
    the liquid is reset to clean water; existing bubbles are kept."""
    st.material, st.c_bulk = name, 0.0
    sim.set_material(name, 0.0, st.wall)
    sim.fill_c(0.0)
    st.dirty = True


def set_bulk(st, value):
    """Change the bulk concentration. The whole surface is shifted by the
    difference at once, which keeps any drops and gradients already on it;
    waiting for the slow bulk exchange would make the slider feel dead."""
    if value != st.c_bulk:
        sim.shift_c(value - st.c_bulk)
        st.c_bulk = value
        st.dirty = True


def panel(gui, st, stats):
    """Draw the two overlay windows and apply what the user changed.

    Window positions and sizes are fractions of the whole window. The labels
    are English because the built-in UI font has no CJK glyphs.
    """
    m = sim.MATERIALS[st.material]
    i = st.info
    g = sim.gamma_host(st.material, st.c_bulk)
    with gui.sub_window("Side view: bubble shape (dashed = pure water)",
                        sim.VIEW / sim.W + 0.005, 0.0, sim.PANEL / sim.W - 0.01, 0.075):
        gui.text(f"Bond {i['bond']:.2f}   cap radius {i['rc']:.2f} mm   contact angle {i['theta']:.0f} deg")
        gui.text(f"{i['height']:.2f} mm above the surface, {i['depth']:.2f} mm below")
    with gui.sub_window("Controls", sim.VIEW / sim.W + 0.005, 0.34, sim.PANEL / sim.W - 0.01, 0.655):
        gui.text("LMB: add bubbles   RMB: add a drop of the additive")
        gui.text(f"bubbles {stats['n']}   bursts {stats['bursts']}   merges {stats['merges']}"
                 f"   {stats['ms']:.1f} ms on {ti.cfg.arch.name}")
        gui.text(f"Liquid: {st.material}")
        for name in NAMES:
            if name != st.material and gui.button(name):
                select(st, name)
        if m["cmax"] > 0:
            hi = 0.8 if st.material == "Glycerol" else 1.0
            set_bulk(st, gui.slider_float("bulk conc.", st.c_bulk, 0.0, hi))
            gui.text(f"  = {st.c_bulk * m['cmax']:.3g} {m['unit']}")
        gui.text(f"surface tension {g:.1f} mN/m   capillary length {i['lc']:.2f} mm")
        r = gui.slider_float("side-view R (mm)", st.probe_r, 0.2, 7.0)
        if r != st.probe_r:
            st.probe_r, st.dirty = r, True
        st.emit_rate = gui.slider_float("bubbles / s", st.emit_rate, 0.0, 80.0)
        st.mean_r = gui.slider_float("mean R (mm)", st.mean_r, 0.3, 4.0)
        st.drop = gui.slider_float("drop strength", st.drop, 0.1, 3.0)
        st.speed = gui.slider_float("time scale", st.speed, 0.1, 2.0)
        w = gui.checkbox("wall attracts bubbles", st.wall)
        if w != st.wall:
            st.wall, st.dirty = w, True
        st.show_gamma = gui.checkbox("colour by surface tension", st.show_gamma)
        st.paused = gui.checkbox("pause (space)", st.paused)
        if gui.button("clear bubbles (c)"):
            sim.clear_bubbles()
        if gui.button("reset liquid (r)"):
            sim.fill_c(st.c_bulk)


def run(shot=None):
    """Main loop. With `shot`, run a scripted 240 frames, save a screenshot and exit."""
    st = State()
    window = ti.ui.Window("Bubbles on a liquid surface", (sim.W, sim.H), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    select(st, "Pure water")
    apply(st)
    stats = dict(n=0, bursts=0, merges=0, ms=0.0)
    carry = 0.0          # fractional bubbles owed by the emitter
    last = time.perf_counter()
    frame = 0
    if shot:
        select(st, "Soap (SDS)")
        set_bulk(st, 0.3)
        sim.fill_c(0.3)
        st.emit_rate = 60.0

    while window.running:
        frame += 1
        # Mouse input is paced by real time, the model by a fixed step: a slow
        # frame then slows the simulation down instead of making it unstable.
        now = time.perf_counter()
        real_dt, last = min(now - last, 0.05), now
        dt = (1.0 / 60.0) * st.speed

        if shot:                                  # scripted, ignores input
            if frame == 200:
                sim.add_drop(35.0, 60.0, 1.2, 3.0)
        else:
            for e in window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.ESCAPE:
                    window.running = False
                elif e.key == ti.ui.SPACE:
                    st.paused = not st.paused
                elif e.key == "c":
                    sim.clear_bubbles()
                elif e.key == "r":
                    sim.fill_c(st.c_bulk)
            # Cursor is in [0, 1]^2 with the origin at the bottom left, the
            # same orientation as the image; convert to tank millimetres and
            # ignore it while it is over the side panel.
            mx, my = window.get_cursor_pos()
            in_view = mx * sim.W < sim.VIEW
            wx, wy = mx * sim.W / sim.VIEW * sim.L, my * sim.L
            if in_view and window.is_pressed(ti.ui.LMB):
                n = max(1, int(60 * real_dt))
                sim.spawn_bubbles(sim.random_xy(n, (wx, wy), 2.5), sim.random_radii(n, st.mean_r))
            if in_view and window.is_pressed(ti.ui.RMB) and sim.MATERIALS[st.material]["cmax"] > 0:
                sim.add_drop(wx, wy, st.drop * real_dt * 8.0, 2.5)

        if st.dirty:
            apply(st)
        if not st.paused:
            # Emitter: bubbles appear at random places at the chosen rate.
            carry += st.emit_rate * dt
            if carry >= 1.0:
                n = int(carry)
                carry -= n
                sim.spawn_bubbles(sim.random_xy(n), sim.random_radii(n, st.mean_r))
            t0 = time.perf_counter()
            b, mg = sim.step(dt)
            stats["bursts"] += b
            stats["merges"] += mg
            stats["n"] = sim.count_alive()
            stats["ms"] = 0.9 * stats["ms"] + 0.1 * (time.perf_counter() - t0) * 1e3

        # The frame is drawn on the GPU into one image; the UI goes on top.
        sim.render(int(st.show_gamma))
        canvas.set_image(sim.img)
        panel(gui, st, stats)
        if shot and frame == 240:
            window.save_image(shot)
            break
        window.show()


def test(out):
    """Scripted run: every material, bubbles + a drop in the middle, with stats."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # the GPU Bessel approximation against SciPy, on both branches (x <= 2, x > 2)
    from scipy.special import k1
    xs = [0.05, 0.5, 1.0, 2.0, 2.5, 6.0]
    err = max(abs(sim.k1_probe(x) / k1(x) - 1) for x in xs)
    print(f"arch={ti.cfg.arch.name}  Bessel K1 max relative error {err:.1e}")

    st = State()
    dt = 1 / 60
    for name, c in [("Pure water", 0.0), ("Soap (SDS)", 0.0), ("Soap (SDS)", 0.6),
                    ("Ethanol", 0.3), ("Salt (NaCl)", 0.5), ("Glycerol", 0.6)]:
        select(st, name)
        set_bulk(st, c)
        sim.fill_c(c)
        apply(st)
        sim.clear_bubbles()
        sim.spawn_bubbles(sim.random_xy(400), sim.random_radii(400, 1.2))
        bursts = merges = 0
        t0 = time.perf_counter()
        log = []
        for f in range(600):                       # 10 s of model time
            if f == 240 and sim.MATERIALS[name]["cmax"] > 0:
                sim.add_drop(50.0, 50.0, 1.2, 3.0)
            b, mg = sim.step(dt)
            bursts, merges = bursts + b, merges + mg
            if f in (59, 239, 270, 599):
                log.append(f"t={(f + 1) / 60:.1f}s n={sim.count_alive()}")
            if f in (120, 270):
                sim.render(0)
                tag = f"{name.split()[0].lower()}_{c:g}_t{(f + 1) / 60:.1f}"
                ti.tools.imwrite(sim.img, str(out / f"{tag}.png"))
        ms = (time.perf_counter() - t0) / 600 * 1e3
        # a blown-up run shows as NaN or infinity in the fields
        cf = sim.c0.to_numpy()
        ok = np.isfinite(cf).all() and np.isfinite(sim.pos.to_numpy()).all()
        i = st.info
        print(f"{name:12s} c={c:<4g} gamma={sim.gamma_host(name, c):5.1f} lc={i['lc']:.2f} "
              f"| {'  '.join(log)} | bursts={bursts} merges={merges} "
              f"| c range {cf.min():.3f}..{cf.max():.3f} finite={ok} | {ms:.1f} ms/step")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        test(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "--shot":
        run(sys.argv[2])
    else:
        run()
