#!/usr/bin/env python3
"""
X-ray source / geometry tradeoff explorer -- tabletop PCB CT
============================================================

Live-slider design tool for picking an X-ray source + geometry against the
project requirements:

    Must   : ~50 µm effective resolution  (0.5 mm pitch BGA/QFN, 0.15 mm traces)
    Should : ~25-30 µm                    (0.4 mm pitch BGA, large voids)
    May    : ~10-15 µm                    (0.25 mm pitch BGA, microfocus)

Panels
------
1. Resolution vs magnification at the current power (spot + pixel blur terms).
2. DESIGN MAP: magnification (x) vs source power (y), coloured by resolution.
   Requirement tiers are contours, relative exposure time is the right axis,
   geometry limits are shaded.  This is the "sweet spot" overlay.
3. Polychromatic transmission vs thickness (Cu, solder, FR4) at current kV.
4. Readout of the operating point (set by FDD, FOD and P sliders; M = FDD/FOD).

Model (first-order, fast -- the browser simulator is the detailed check)
------------------------------------------------------------------------
spot            s   = max(s_min, P / k)                [µm, k in W/µm]
resolution      r   = sqrt( (s*(1-1/M))^2 + (p/M)^2 )   [µm at object]
magnification   M   = FDD / FOD   (both measured from the focal spot)
FOD floor       FOD_min = (W/2)*sin(tilt_max) + clearance + spot-to-window
                (board perpendicular to beam, rotation axis in board plane)
field of view   FOV = detector width / M
tube output     flux ∝ I*kV^2 = P*kV
exposure time   t ∝ FDD^2 / (P * kV * T_stack)   (relative to 90 kV, 8 W, FDD 300)
spectrum        Kramers bremsstrahlung, N(E) ∝ (kVp-E)/E, energy-integrating
                detector, 10 keV cutoff (window/air). No char. lines, scatter,
                heel effect, or detector efficiency vs energy.

Attenuation tables are approximate NIST XCOM values (μ/ρ, cm²/g). Solder is
treated as Sn (SAC305 is ~96.5 % Sn); FR4 as a glass/epoxy blend. Check them
against XCOM before you quote numbers in the report.

Run:  python xray_tradeoff_explorer.py   (needs numpy + matplotlib)
In Spyder, set Graphics backend to "Automatic"/Qt so the sliders are live.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# --------------------------------------------------------------------------
# Requirement tiers: (name, band_low_um, pass_threshold_um, colour)
# --------------------------------------------------------------------------
TIERS = [
    ("Must",   50, 50, "#d62728"),
    ("Should", 25, 30, "#ff7f0e"),
    ("May",    10, 15, "#2ca02c"),
]

# --------------------------------------------------------------------------
# Mass attenuation coefficients μ/ρ [cm²/g] (approx. NIST XCOM, total w/ coherent)
# --------------------------------------------------------------------------
MATS = {
    "Cu": dict(rho=8.96, color="#b8732e",
               E=[10, 15, 20, 30, 40, 50, 60, 80, 100, 150],
               mu=[215.9, 74.05, 33.79, 10.92, 4.862, 2.613, 1.593, 0.763, 0.4584, 0.2217]),
    "Solder": dict(rho=7.40, color="#7f7f7f",   # ~Sn, K-edge at 29.2 keV
               E=[10, 15, 20, 25, 29.19, 29.21, 30, 40, 50, 60, 80, 100, 150],
               mu=[136.1, 44.5, 20.3, 11.3, 7.8, 41.5, 38.5, 18.6, 10.3, 6.35, 2.95, 1.65, 0.62]),
    "FR4": dict(rho=1.85, color="#2ca02c",
               E=[10, 15, 20, 30, 40, 50, 60, 80, 100, 150],
               mu=[20.0, 6.5, 3.0, 1.05, 0.55, 0.37, 0.29, 0.22, 0.19, 0.15]),
}


def mu_lin(mat, E):
    """Linear attenuation coefficient [1/mm] at energies E [keV] (log-log interp)."""
    m = MATS[mat]
    lm = np.interp(np.log(E), np.log(m["E"]), np.log(m["mu"]))
    return np.exp(lm) * m["rho"] / 10.0


def spectrum(kvp):
    """Kramers spectrum weighted for an energy-integrating detector."""
    kvp = max(kvp, 10.5)
    E = np.arange(10.0, kvp, 0.25)
    w = (kvp - E) / E * E          # photon number × energy
    return E, w / w.sum()


def transmission(kvp, layers):
    """Energy-weighted transmission through a stack [(mat, thickness_mm), ...]."""
    E, w = spectrum(kvp)
    att = np.zeros_like(E)
    for mat, t in layers:
        att += mu_lin(mat, E) * t
    return float((w * np.exp(-att)).sum())


def transmission_curve(kvp, mat, t_mm, prefilter=()):
    E, w = spectrum(kvp)
    pre = np.zeros_like(E)
    for m, t in prefilter:
        pre += mu_lin(m, E) * t
    mu = mu_lin(mat, E)
    T = (w[:, None] * np.exp(-pre[:, None] - mu[:, None] * t_mm[None, :])).sum(0)
    return T / (w * np.exp(-pre)).sum()


# --------------------------------------------------------------------------
# Imaging model
# --------------------------------------------------------------------------
def spot_um(P, k, smin):
    return np.maximum(smin, P / k)


def resolution_um(M, P, k, smin, pix):
    s = spot_um(P, k, smin)
    return np.hypot(s * (1 - 1 / M), pix / M)


def power_ceiling(r, M, k, smin, pix):
    """Highest power that still meets resolution r at magnification M (None = impossible)."""
    if M <= 1:
        return np.inf if pix / M <= r else None
    pt = pix / M
    if pt >= r:
        return None
    s_allowed = np.sqrt(r**2 - pt**2) / (1 - 1 / M)
    if s_allowed < smin:
        return None
    return k * s_allowed


REF = dict(P=8.0, kV=90.0, FDD=300.0)   # exposure-time reference point

# --------------------------------------------------------------------------
# Figure + sliders
# --------------------------------------------------------------------------
M_LO, M_HI = 1.0, 200.0
P_LO, P_HI = 0.3, 200.0

fig = plt.figure(figsize=(17, 10.5))
fig.canvas.manager.set_window_title("X-ray tradeoff explorer")
ax_res = fig.add_axes([0.05, 0.55, 0.26, 0.36])
ax_map = fig.add_axes([0.36, 0.55, 0.24, 0.36])
cax    = fig.add_axes([0.665, 0.55, 0.008, 0.36])
ax_T   = fig.add_axes([0.77, 0.55, 0.215, 0.36])

# slider specs: key, label, min, max, init, fmt, log
SPECS = [
    # column 1: source
    ("kV",    "Tube voltage [kV]",           40, 160, 90,   "{:.0f}", False),
    ("P",     "Source power [W]",            P_LO, P_HI, 8, "{:.2f}", True),
    ("k",     "Spot coeff k [W/µm]",         0.3, 3.0, 0.75, "{:.2f}", False),
    ("smin",  "Min spot s_min [µm]",         1, 30, 5,      "{:.1f}", False),
    ("s2w",   "Spot-to-window [mm]",         2, 25, 7.3,    "{:.1f}", False),
    ("cone",  "Cone angle [deg]",            20, 140, 118,  "{:.0f}", False),
    ("FDD",   "FDD (spot→detector) [mm]",    100, 1500, 300, "{:.0f}", False),
    ("FOD",   "FOD (spot→object) [mm]",      2, 1500, 60,   "{:.1f}", True),
    # column 2: detector / object
    ("pix",   "Detector pixel [µm]",         10, 200, 49.5, "{:.1f}", False),
    ("detw",  "Detector width [mm]",         20, 450, 115,  "{:.0f}", False),
    ("W",     "Board width [mm]",            5, 200, 100,   "{:.0f}", False),
    ("tilt",  "Max tilt from face-on [deg]", 0, 90, 60,     "{:.0f}", False),
    ("clr",   "Clearance [mm]",              0, 30, 5,      "{:.1f}", False),
    ("ROI",   "CT ROI width [mm]",           5, 150, 50,    "{:.0f}", False),
    ("brd",   "FR4 thickness [mm]",          0.4, 4.0, 1.6, "{:.2f}", False),
    ("ball",  "Solder path [mm]",            0.05, 1.0, 0.25, "{:.2f}", False),
    ("void",  "Void size [µm]",              5, 150, 27,    "{:.0f}", False),
]

sliders = {}
for i, (key, label, lo, hi, init, fmt, islog) in enumerate(SPECS):
    col, row = (0, i) if i < 9 else (1, i - 9)
    x0 = 0.11 + col * 0.31
    y0 = 0.455 - row * 0.05
    sax = fig.add_axes([x0, y0, 0.18, 0.022])
    if islog:
        s = Slider(sax, label, np.log10(lo), np.log10(hi), valinit=np.log10(init))
        s.valtext.set_text(fmt.format(init))
    else:
        s = Slider(sax, label, lo, hi, valinit=init)
        s.valtext.set_text(fmt.format(init))
    s.label.set_fontsize(8.5)
    s.valtext.set_fontsize(8.5)
    sliders[key] = (s, fmt, islog)


def val(key):
    s, _, islog = sliders[key]
    return 10 ** s.val if islog else s.val


readout = fig.text(0.655, 0.475, "", family="monospace", fontsize=8.6, va="top")

norm = LogNorm(vmin=3, vmax=300)
cmap = plt.get_cmap("viridis")
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("effective resolution [µm]")

STATE = {"t_coef": 1.0, "sec": None, "FDD": 300.0, "top": []}


def fod_axis(ax):
    """Top x-axis showing FOD [mm] for the current FDD (FOD = FDD / M)."""
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda m: STATE["FDD"] / np.maximum(m, 1e-9),
                   lambda f: STATE["FDD"] / np.maximum(f, 1e-9)))
    top.set_xlabel("FOD, spot→object [mm]", fontsize=8)
    top.tick_params(labelsize=8)
    STATE["top"].append(top)

Mg = np.logspace(np.log10(M_LO * 1.001), np.log10(M_HI), 220)
Pg = np.logspace(np.log10(P_LO), np.log10(P_HI), 220)
MM, PP = np.meshgrid(Mg, Pg)


def shade_limits(ax, M_geom, M_fov, ytxt):
    if M_geom < M_HI:
        ax.axvspan(M_geom, M_HI, color="0.3", alpha=0.25, hatch="//", lw=0)
        ax.axvline(M_geom, color="k", lw=1.2)
        ax.text(M_geom * 1.06, ytxt, "board hits window →", fontsize=7.5,
                rotation=90, va="bottom", ha="left")
    ax.axvline(M_fov, color="k", lw=1.2, ls="--")
    ax.text(M_fov * 0.94, ytxt, "ROI fits one frame ←", fontsize=7.5,
            rotation=90, va="bottom", ha="right")


def update(_=None):
    for key, (s, fmt, islog) in sliders.items():
        s.valtext.set_text(fmt.format(val(key)))

    kV, P, k, smin = val("kV"), val("P"), val("k"), val("smin")
    s2w, cone, FOD, FDD = val("s2w"), val("cone"), val("FOD"), val("FDD")
    STATE["FDD"] = FDD
    fod_bad = FOD >= FDD
    M = max(FDD / FOD, 1.0)
    pix, detw, W, tilt = val("pix"), val("detw"), val("W"), val("tilt")
    clr, ROI, brd, ball, void = val("clr"), val("ROI"), val("brd"), val("ball"), val("void")

    # ---- geometry ------------------------------------------------------
    FOD_min = (W / 2) * np.sin(np.radians(tilt)) + clr + s2w
    M_geom = FDD / FOD_min
    M_fov = max(detw / ROI, 1.0)
    FOV = detw / M
    det_half = np.degrees(np.arctan((detw / 2) / FDD)) * 2
    beam_at_obj = 2 * FOD * np.tan(np.radians(cone / 2))

    # ---- physics -------------------------------------------------------
    s = float(spot_um(P, k, smin))
    r = float(resolution_um(M, P, k, smin, pix))
    I_uA = P / kV * 1000
    T_stack = transmission(kV, [("FR4", brd), ("Solder", ball)])
    T_void = transmission(kV, [("FR4", brd), ("Solder", max(ball - void / 1000, 0))])
    c_void = (T_void - T_stack) / T_stack
    T_brd = transmission(kV, [("FR4", brd)])
    c_trace = 1 - transmission(kV, [("FR4", brd), ("Cu", 0.035)]) / T_brd
    T_ref = transmission(REF["kV"], [("FR4", brd), ("Solder", ball)])
    t_coef = (FDD**2 / (kV * T_stack)) / (REF["FDD"]**2 / (REF["P"] * REF["kV"] * T_ref))
    STATE["t_coef"] = t_coef          # t_rel(P) = t_coef / P
    t_rel = t_coef / P

    # ---- panel 1: resolution vs M --------------------------------------
    for t in STATE["top"]:
        t.remove()
    STATE["top"] = []
    ax_res.cla()
    for name, lo, hi, c in TIERS:
        ax_res.axhspan(lo, hi if hi > lo else lo * 1.04, color=c, alpha=0.25)
        ax_res.text(1.05, hi * 1.05, name, color=c, fontsize=8)
    for Pc, a in [(k * smin, 0.35), (P_HI / 2, 0.35)]:
        ax_res.plot(Mg, resolution_um(Mg, Pc, k, smin, pix), color="0.5", lw=0.8, alpha=a)
    ax_res.plot(Mg, s * (1 - 1 / Mg), "--", color="#9467bd", lw=1, label="spot blur")
    ax_res.plot(Mg, pix / Mg, "--", color="#1f77b4", lw=1, label="pixel blur")
    ax_res.plot(Mg, resolution_um(Mg, P, k, smin, pix), color="k", lw=2,
                label=f"total @ {P:.1f} W")
    ax_res.plot(M, r, "o", color="r", ms=7, zorder=5)
    shade_limits(ax_res, M_geom, M_fov, 2.3)
    ax_res.set(xscale="log", yscale="log", xlim=(M_LO, M_HI), ylim=(2, 300),
               xlabel="magnification M = FDD / FOD", ylabel="effective resolution at object [µm]")
    ax_res.set_title("Resolution vs magnification", pad=26, fontsize=10)
    fod_axis(ax_res)
    ax_res.grid(True, which="both", alpha=0.2)
    ax_res.legend(fontsize=8, loc="upper right")

    # ---- panel 2: design map -------------------------------------------
    if STATE["sec"] is not None:
        STATE["sec"].remove()
    ax_map.cla()
    R = resolution_um(MM, PP, k, smin, pix)
    ax_map.contourf(MM, PP, R, levels=np.logspace(np.log10(3), np.log10(300), 40),
                    cmap=cmap, norm=norm, extend="both")
    for name, lo, hi, c in TIERS:
        cs = ax_map.contour(MM, PP, R, levels=[hi], colors=[c], linewidths=2.2)
        ax_map.clabel(cs, fmt={hi: f"{name} {hi} µm"}, fontsize=8)
    ax_map.axhline(k * smin, color="w", lw=0.8, ls=":")
    ax_map.text(M_LO * 1.1, k * smin * 1.08, "spot floor", color="w", fontsize=7.5)
    shade_limits(ax_map, M_geom, M_fov, P_LO * 1.3)
    ax_map.plot(M, P, "o", color="r", mec="w", ms=9, zorder=5)
    ax_map.set(xscale="log", yscale="log", xlim=(M_LO, M_HI), ylim=(P_LO, P_HI),
               xlabel="magnification M = FDD / FOD", ylabel="source power [W]")
    ax_map.set_title("Design map (sweet spot overlay)", pad=26, fontsize=10)
    fod_axis(ax_map)
    sec = ax_map.secondary_yaxis(
        "right",
        functions=(lambda p: STATE["t_coef"] / np.maximum(p, 1e-9),
                   lambda t: STATE["t_coef"] / np.maximum(t, 1e-9)))
    sec.set_ylabel("relative exposure time (×)", fontsize=8)
    STATE["sec"] = sec

    # ---- panel 3: transmission -----------------------------------------
    ax_T.cla()
    tt = np.linspace(0.0, 3.0, 120)
    for mat in ("FR4", "Solder", "Cu"):
        ax_T.plot(tt, transmission_curve(kV, mat, tt), color=MATS[mat]["color"], lw=2, label=mat)
    ax_T.plot(tt, transmission_curve(kV, "Solder", tt, prefilter=[("FR4", brd)]),
              color=MATS["Solder"]["color"], ls="--", lw=1.2, label=f"Solder behind {brd:.1f} mm FR4")
    ax_T.axvline(brd, color=MATS["FR4"]["color"], ls=":", lw=1)
    ax_T.axvline(ball, color=MATS["Solder"]["color"], ls=":", lw=1)
    ax_T.axhspan(1e-3, 0.02, color="r", alpha=0.08)
    ax_T.text(2.0, 0.006, "photon-starved", color="r", fontsize=8)
    ax_T.set(yscale="log", ylim=(1e-3, 1.05), xlim=(0, 3),
             xlabel="thickness [mm]", ylabel="transmission (energy-weighted)",
             title=f"Transmission at {kV:.0f} kVp")
    ax_T.grid(True, which="both", alpha=0.2)
    ax_T.legend(fontsize=8, loc="upper right")

    # ---- readout -------------------------------------------------------
    tiers_txt = []
    for name, lo, hi, _ in TIERS:
        ok = "PASS" if r <= hi else "fail"
        pc = power_ceiling(hi, min(M, M_geom), k, smin, pix)
        pc_txt = "not reachable" if pc is None else ("any" if np.isinf(pc) else f"≤ {pc:6.1f} W")
        tiers_txt.append(f"  {name:<6} ≤{hi:>3} µm : {ok:<4}  power ceiling {pc_txt}")
    geom_ok = "OK" if FOD >= FOD_min else "BLOCKED: board hits window"
    if fod_bad:
        geom_ok = "FOD ≥ FDD: object behind detector!"
    fov_ok = "fits" if FOV >= ROI else f"needs ~{int(np.ceil(ROI / FOV))}× (stitch)"
    cone_ok = "OK" if det_half <= cone else "detector wider than cone"
    I_flag = "" if I_uA <= 500 else "  (high for a microfocus tube)"

    readout.set_text("\n".join([
        "OPERATING POINT  (red dot)",
        f"  spot            {s:6.1f} µm      P/k floor {k * smin:5.1f} W",
        f"  resolution      {r:6.1f} µm      (spot {s * (1 - 1 / M):4.1f}, pixel {pix / M:4.1f})",
        f"  tube current    {I_uA:6.0f} µA{I_flag}",
        "",
        "GEOMETRY",
        f"  M = FDD/FOD     {M:6.2f}×      ({FDD:.0f} / {FOD:.1f} mm)",
        f"  FOD             {FOD:6.1f} mm  (window gap {FOD - s2w:5.1f} mm)",
        f"  FOD min         {FOD_min:6.1f} mm  → M max {M_geom:5.2f}   [{geom_ok}]",
        f"  FOV at object   {FOV:6.1f} mm  vs ROI {ROI:.0f} mm: {fov_ok}",
        f"  beam at object  {beam_at_obj:6.0f} mm wide;  detector needs {det_half:4.0f}° cone [{cone_ok}]",
        "",
        "REQUIREMENTS (ceiling evaluated at min(M, M max))",
        *tiers_txt,
        "",
        "CONTRAST / DOSE",
        f"  T through FR4+solder   {T_stack:6.3f}",
        f"  {void:.0f} µm void contrast    {100 * c_void:5.1f} %",
        f"  35 µm Cu trace contrast {100 * c_trace:5.1f} %",
        f"  rel. exposure time      {t_rel:6.2f}×  (ref 90 kV, 8 W, FDD 300)",
    ]))
    fig.canvas.draw_idle()


for s, _, _ in sliders.values():
    s.on_changed(update)

bax = fig.add_axes([0.9, 0.02, 0.07, 0.035])
btn = Button(bax, "Reset")
btn.on_clicked(lambda _: [s.reset() for s, _, _ in sliders.values()])

update()

if __name__ == "__main__":
    plt.show()