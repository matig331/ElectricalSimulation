"""paper_fig_readouts.py -- publication figure: one soma Vm trace and what is read out of it.

Three panels, as little text as possible (the caption carries the explanation):

  A  soma Vm, stimulated and sham, from 100 ms before to 800 ms after the pulse. The area
     between them is DeltaV; the dashed line is the fitted reconstruction. The sub-ms
     transient at the pulse is clipped and boxed; the box is enlarged in B.
  B  the direct polarisation, sub-millisecond: the biphasic current above; DeltaV_end at the
     end of phase 2 (t0); the post-pulse peak DeltaV_pk; the single exponential (tau_m)
     fitted from that peak; the model-free t_1/e.
  C  the bump alone, read out the way B reads out the direct polarisation: DeltaV minus
     the fast term, which starts from 0 at t0 -- the trace's value there, DeltaV_end, is
     taken as the bump's zero (the pinning; squares and arrow). The pinned double
     exponential with its two terms, +A_b exp(-u/tau_d) and -A_b exp(-u/tau_r); tau_d and
     tau_r as dimension lines from t0 to where each term has fallen to 1/e (as t_1/e in B);
     t_b from t0 to the peak; DeltaV_b from the zero to the peak.

Every number drawn comes from the SAME code the campaign uses (plot_vm_examples.analyse ->
bump_kinetics.fit_staged), so the figure cannot illustrate a method the pipeline does not
run.

PIPELINE -- one concern per section; only section 1 needs NEURON
  1. acquisition  simulate one placement -> .npz (traces + metadata)     cluster
  2. readouts     the fits, and every coordinate the figure draws        numpy
  3. style        palette, font, rcParams
  4. panels       one function per panel; draws, computes nothing
  5. layout       fixed millimetre geometry, key, panel letters
  6. export       PDF (TrueType embedded), SVG (text stays text), PNG 600 dpi
  7. checks       text collisions and text outside the page

USAGE
  # on the cluster (NEURON): simulate a placement and draw it
  python paper_fig_readouts.py all --placement 130303:120:-110:-110:270 --out-stem fig_readouts

  # a neuron a campaign simulated, picked from campaign_examples.csv (example number 1..N)
  python paper_fig_readouts.py all --from-examples-csv analysis/<dir>/campaign_examples.csv \
         --example 7 --out-stem fig_readouts

  # redraw from saved data anywhere (laptop): numpy + matplotlib, no NEURON
  python paper_fig_readouts.py plot --data fig_readouts.npz --out-stem fig_readouts

  # which neuron? rank the rows of an examples CSV (both fits accepted, clear bump, ...)
  python paper_fig_readouts.py suggest --from-examples-csv analysis/<dir>/campaign_examples.csv

  --width-mm 180 (double column, default) or e.g. 120; --font to force a family.

  Outputs <stem>.pdf / .svg / .png, <stem>.npz (the data) and <stem>_values.txt (every
  number read out, for the caption).

Test:  python smoke_test_paper_fig_readouts.py      (no NEURON, ~10 s)
"""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib import font_manager                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TRACE_KEYS = ("t_fine", "v_fine", "t_fine_sham", "v_fine_sham", "t_dv_fine", "dv_fine",
              "t_grid", "v_grid", "v_grid_sham", "dv")
MM = 1.0 / 25.4                                                   # inches per millimetre


# ===========================================================================================
# 1. acquisition -- the only part that needs NEURON
# ===========================================================================================
def parse_placement(text):
    """'morph:layer:x:y:theta' -> dict. Same syntax as plot_vm_examples --placements."""
    parts = str(text).strip().split(":")
    if len(parts) != 5:
        raise ValueError("placement must be morph:layer:x:y:theta, got %r" % (text,))
    return dict(morph=parts[0], layer=float(parts[1]), x_um=float(parts[2]),
                y_um=float(parts[3]), theta_deg=float(parts[4]), i0_uA=None)


def placement_from_examples_csv(path, example):
    """One row of plot_vm_examples.csv or campaign_examples.csv -> placement dict.

    `example` is the 'example' column of campaign_examples.csv (1..N), or, for
    plot_vm_examples.csv which has no such column, the 1-based row number. The CSV carries
    the EXACT position and raw rotation, so the neuron is reproduced as simulated there.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("%s has no rows" % path)
    if "example" in rows[0]:
        hit = [r for r in rows if int(float(r["example"])) == int(example)]
    else:
        hit = rows[int(example) - 1:int(example)] if 1 <= int(example) <= len(rows) else []
    if not hit:
        raise SystemExit("example %s not found in %s (%d rows)" % (example, path, len(rows)))
    r = hit[0]
    return dict(morph=r["morph"], layer=float(r["layer_um"]), x_um=float(r["x_um"]),
                y_um=float(r["y_um"]), theta_deg=float(r["theta_deg"]),
                i0_uA=(float(r["i0_uA"]) if r.get("i0_uA", "") != "" else None))


def _f(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return float("nan")


def suggest_examples(path, top=8):
    """Rank the rows of plot_vm_examples.csv / campaign_examples.csv by how well each would
    illustrate the method, best first -> list of (score, example, row).

    Only rows where BOTH fits were accepted and the cell did not fire qualify. The score
    favours a large bump with a clean fit (r2), a post-pulse peak that is NOT at t0 (so
    DeltaV_end and DeltaV_pk are two visible points, as the method distinguishes them), and
    a fast response comparable to the bump rather than dwarfing it. It ranks; it does not
    decide -- look at the candidates' pages before choosing.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ranked = []
    for i, r in enumerate(rows):
        if _f(r, "dexp_fit_ok") != 1 or _f(r, "early_fit_ok") != 1 or _f(r, "fired") == 1:
            continue
        bump, r2 = abs(_f(r, "dexp_peak_mV")), _f(r, "dexp_r2")
        fast = abs(_f(r, "early_peak_mV"))
        if not (np.isfinite(bump) and np.isfinite(r2) and np.isfinite(fast)) or bump <= 0:
            continue
        distinct = _f(r, "early_t_peak_ms") >= 0.05
        ratio = fast / bump
        score = (bump * (r2 ** 20) * (1.0 if distinct else 0.6)
                 * (1.0 if 0.3 <= ratio <= 3.0 else 0.5))
        ex = int(_f(r, "example")) if "example" in r else i + 1
        ranked.append((float(score), ex, r))
    ranked.sort(key=lambda z: -z[0])
    return ranked[:int(top)]


def acquire(placement, cell_model="full_tuned", bump_ms=800.0, baseline_ms=100.0):
    """Simulate one placement with the campaign's protocol -> (rec, meta).

    rec holds the traces (TRACE_KEYS); meta everything needed to redraw them without NEURON:
    the protocol, the stimulus waveform parameters actually simulated, the fit settings.
    """
    import inspect
    from config import CFG
    import plot_vm_examples as P
    import rich_footprint

    proto = P.default_protocol(
        CFG, i0_uA=placement.get("i0_uA"), bump_ms=float(bump_ms),
        bump_dt_ms=float(getattr(CFG, "bump_dt_ms", 0.5)),
        cvode_atol=float(getattr(CFG, "cvode_atol", 1e-6)),
        play_margin_ms=float(getattr(CFG, "play_margin_ms", 3.0)),
        baseline_ms=float(baseline_ms))
    inst = P.build_instance(placement["morph"], placement["layer"], cell_model, cfg=CFG,
                            tag="_pfr_")
    sham = P.sham_reference(inst, proto)
    rec = P.simulate_instance(inst, proto, sham, (placement["x_um"], placement["y_um"]),
                              placement["theta_deg"])
    # the waveform spikes_at actually plays: its own ramp/interphase defaults, anodic first
    sig = inspect.signature(rich_footprint.spikes_at).parameters
    meta = dict(
        morph=str(inst["morph"]), layer_um=float(inst["layer"]), cell_model=str(cell_model),
        v_rest_mV=float(inst["v_rest"]), x_um=float(placement["x_um"]),
        y_um=float(placement["y_um"]), theta_deg=float(placement["theta_deg"]),
        r_um=float(rec["r_um"]), fired=int(rec["fired"]),
        i0_uA=float(proto["i0_uA"]), phase_dur_ms=float(proto["phase_dur_ms"]),
        ramp_us=float(sig["ramp_us"].default), interphase_us=float(sig["interphase_us"].default),
        anodic_first=True, dt_ms=float(proto["dt_ms"]), bump_ms=float(proto["bump_ms"]),
        bump_dt_ms=float(proto["bump_dt_ms"]), play_margin_ms=float(proto["play_margin_ms"]),
        baseline_ms=float(proto["baseline_ms"]),
        early_floor=float(getattr(CFG, "bump_early_floor", 0.25)),
        t0_ms=float(getattr(CFG, "bump_t0_ms", 0.0)),
        created=datetime.datetime.now().isoformat(timespec="seconds"))
    return {k: np.asarray(rec[k], float) for k in TRACE_KEYS}, meta


def save_record(path, rec, meta):
    """Traces + metadata in one .npz (metadata as a JSON string; no pickle needed to load)."""
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(path, meta_json=np.array(json.dumps(meta, sort_keys=True)),
                        **{k: np.asarray(rec[k], float) for k in TRACE_KEYS})
    return path


def load_record(path):
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in TRACE_KEYS + ("meta_json",) if k not in z.files]
        if missing:
            raise SystemExit("%s lacks %s -- not written by save_record" % (path, missing))
        meta = json.loads(str(z["meta_json"]))
        rec = {k: np.array(z[k], float) for k in TRACE_KEYS}
    return rec, meta


# ===========================================================================================
# 2. readouts -- the fits, and every coordinate the figure draws (pure numpy)
# ===========================================================================================
def stimulus_waveform(t_ms, meta):
    """The biphasic current (uA) against t measured from the END of phase 2.

    field.biphasic_current is the function the simulation plays; it counts time from the
    pulse onset, which is 2 phases (plus any interphase gap) before the end of phase 2.
    """
    import field as F
    phi = float(meta["phase_dur_ms"])
    t_on = -(2.0 * phi + float(meta.get("interphase_us", 0.0)) * 1e-3)
    return 1e6 * F.biphasic_current(np.asarray(t_ms, float) - t_on, float(meta["i0_uA"]),
                                    phi, bool(meta.get("anodic_first", True)),
                                    float(meta.get("ramp_us", 0.0)),
                                    float(meta.get("interphase_us", 0.0)))


def readouts(rec, meta):
    """Fit one record exactly as the campaign does and return everything the panels draw.

    Returns a dict q. Scalars that are READ OUT (the CSV quantities) are under q["values"];
    the rest are arrays and coordinates, so the drawing layer has nothing left to compute.
    """
    from plot_vm_examples import analyse
    from bump_kinetics import eval_exp_decay

    t0 = float(meta.get("t0_ms", 0.0))
    fit = analyse(rec, mode="staged", t0_ms=t0,
                  early_floor=float(meta.get("early_floor", 0.25)))
    e, b = fit["early"], fit["bump"]
    margin = float(meta["play_margin_ms"])
    q = dict(meta=dict(meta), early=e, bump=b)

    # ---- A: stitched traces. Fixed-dt samples up to the play margin, the uniform grid after
    #      it (past the margin the solver steps are sparse and irregular). -------------------
    tf, vf = rec["t_fine"], rec["v_fine"]
    tfs, vfs = rec["t_fine_sham"], rec["v_fine_sham"]
    tg, vg, vgs, dvg = rec["t_grid"], rec["v_grid"], rec["v_grid_sham"], rec["dv"]
    tdf, dvf = rec["t_dv_fine"], rec["dv_fine"]
    k = tdf.size                           # the common fixed-dt prefix of stim and sham
    m1, m2 = tf < margin, tg >= margin
    q["A_t"] = np.concatenate([tf[m1], tg[m2]])
    q["A_v"] = np.concatenate([vf[m1], vg[m2]])
    m1s = tfs < margin
    q["A_ts"] = np.concatenate([tfs[m1s], tg[m2]])
    q["A_vs"] = np.concatenate([vfs[m1s], vgs[m2]])
    # reconstruction = sham + fast term + bump, from t0 on (nan before: the model starts there)
    mf = (tdf >= t0 - 1e-9) & (tdf < margin)
    q["A_tm"] = np.concatenate([tdf[mf], tg[m2]])
    q["A_vm"] = np.concatenate([vfs[:k][mf] + fit["resid_fine"][mf] + fit["bump_fine"][mf],
                                vgs[m2] + fit["model_grid"][m2]])
    tdp = b.get("data_t_peak_ms", np.nan)
    if np.isfinite(tdp):
        q["A_dv_arrow"] = (float(tdp), float(np.interp(tdp, tg, vgs)),
                           float(np.interp(tdp, tg, vg)))
    else:
        q["A_dv_arrow"] = None

    # ---- B: the direct polarisation on the exact 0.025 ms DeltaV -----------------------------
    t1e, tpk = e.get("t_1e_ms", np.nan), e.get("t_peak_ms", np.nan)
    span = 4.0 * t1e if np.isfinite(t1e) else 1.5
    x_hi = float(min(margin, max(tpk if np.isfinite(tpk) else 0.0, 0.0) + max(span, 1.0)))
    q["B_xlim"] = (-2.0 * float(meta["phase_dur_ms"]) - 0.1, x_hi)
    mb = (tdf >= q["B_xlim"][0]) & (tdf <= x_hi)
    q["B_t"], q["B_dv"] = tdf[mb], dvf[mb]
    tI = np.linspace(q["B_xlim"][0], x_hi, 2400)
    q["B_It"], q["B_I"] = tI, stimulus_waveform(tI, meta)
    q["B_end"] = (t0, float(b["dv_t0_mV"]))
    q["B_peak"] = (float(tpk), float(e["peak_mV"]))
    if np.isfinite(e.get("tau_ms", np.nan)):
        tt = np.linspace(tpk, x_hi, 400)
        q["B_fit_t"], q["B_fit_v"] = tt, eval_exp_decay(tt, e)
        q["B_fit_hi"] = float(e["t_fit_hi_ms"])
    else:
        q["B_fit_t"] = q["B_fit_v"] = None
        q["B_fit_hi"] = np.nan
    q["B_t1e"] = ((float(tpk), float(tpk + t1e), float(e["peak_mV"]) / np.e)
                  if np.isfinite(t1e) else None)

    # ---- C: the bump alone, on the same linear time as it was fitted -----------------------
    # Data = DeltaV minus the fast term the bump fit used, so it starts from 0 at t0.
    t_lo = float(b.get("t_fit_lo_ms", t0))
    post = tg >= t_lo - 1e-9
    u = tg[post] - t_lo
    q["C_u"] = u
    q["C_data"] = dvg[post] - fit["resid_grid"][post]
    q["C_fit"] = fit["bump_grid"][post]
    # The first ~5 tau_m after t0 still hold fast dynamics the fast term does not model (the
    # response can keep growing past t0 before it relaxes, as in B); at C's time scale they
    # would only draw a sub-millisecond hook at t0.
    tau_off = b.get("tau_offset_ms", np.nan)
    q["C_show"] = u >= (5.0 * float(tau_off) if np.isfinite(tau_off) else 0.0)
    q["C_end"] = (0.0, float(b["dv_t0_mV"]))
    amp = b.get("amp_mV", np.nan)
    tr, td = b.get("tau_rise_ms", np.nan), b.get("tau_decay_ms", np.nan)
    if all(np.isfinite([amp, tr, td])):
        uu = np.linspace(0.0, float(u[-1]), 1600)
        q["C_term_u"] = uu
        q["C_term_d"] = amp * np.exp(-uu / td)             # + A_b exp(-u/tau_d)
        q["C_term_r"] = -amp * np.exp(-uu / tr)            # - A_b exp(-u/tau_r)
        q["C_tau_d"] = (float(td), float(amp) / np.e)      # each at 1/e of its value at t0
        q["C_tau_r"] = (float(tr), -float(amp) / np.e)
    else:
        q["C_term_u"] = q["C_term_d"] = q["C_term_r"] = q["C_tau_d"] = q["C_tau_r"] = None
    q["C_peak"] = (float(b["t_peak_ms"]) - t_lo, float(b["peak_mV"]))

    # fit quality of the full stage-2 model (fast term + bump), once the fast part is over
    mdf = (tdf >= t_lo - 1e-9) & (tdf < margin)
    ud = np.concatenate([tdf[mdf] - t_lo, tg[m2] - t_lo])
    dd = np.concatenate([dvf[mdf], dvg[m2]])
    model = np.concatenate([fit["resid_fine"][mdf] + fit["bump_fine"][mdf],
                            fit["model_grid"][m2]])
    late = ud >= 5.0
    resid_rms = float(np.sqrt(np.nanmean((dd[late] - model[late]) ** 2)))

    # ---- the numbers, as the campaign CSV names them ----------------------------------------
    q["values"] = dict(
        deltaVm_end_phase2_mV=float(b["dv_t0_mV"]),
        early_peak_mV=float(e["peak_mV"]), early_t_peak_ms=float(tpk),
        early_tau_ms=float(e.get("tau_ms", np.nan)), early_t_1e_ms=float(t1e),
        early_r2=float(e.get("r2", np.nan)), early_fit_ok=int(e.get("fit_ok", 0)),
        dexp_peak_mV=float(b["peak_mV"]), dexp_t_peak_ms=float(b["t_peak_ms"]),
        dexp_data_peak_mV=float(b.get("data_peak_mV", np.nan)),
        dexp_tau_rise_ms=float(tr), dexp_tau_decay_ms=float(td), dexp_amp_mV=float(amp),
        dexp_offset_mV=float(b.get("offset_mV", np.nan)),
        dexp_tau_offset_ms=float(b.get("tau_offset_ms", np.nan)),
        dexp_r2=float(b.get("r2", np.nan)), dexp_fit_ok=int(b.get("fit_ok", 0)),
        residual_rms_mV=resid_rms,
        v_rest_mV=float(meta["v_rest_mV"]), fired=int(meta.get("fired", 0)))
    return q


# ===========================================================================================
# 3. style
# ===========================================================================================
# Component colours: Okabe-Ito vermillion / bluish green / blue, validated with the dataviz
# validator (light surface, all pairs): worst CVD dE 11.0 (deutan), worst normal-vision dE
# 18.7, every colour >= 3:1 against white -- all PASS. Line style is a second channel
# everywhere (solid data, dashed sham, bands for fits, dashed/dotted for the two terms).
PAL = dict(ink="#1a1a1a", ink2="#5f5f5f", hair="#c8c8c8", sham="#8c8c8c",
           pulse="#eeeeee", pulse2="#e2e2e2", wave="#7a7a7a",
           fast="#D55E00", slow="#009E73", model="#0072B2")
FONT_PREFERENCE = ("Arial", "Helvetica", "Liberation Sans", "TeX Gyre Heros", "Nimbus Sans",
                   "FreeSans", "DejaVu Sans")
FS = 7.0                                   # base font size (pt) at the printed size


def pick_font(preference=FONT_PREFERENCE):
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in preference:
        if name in have:
            return name
    return "DejaVu Sans"


def paper_rc(font=None):
    fam = font or pick_font()
    return {
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"],
        "font.size": FS, "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": fam + ":italic", "mathtext.bf": fam + ":bold",
        "mathtext.sf": fam, "mathtext.cal": fam, "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.fallback": "stixsans", "mathtext.default": "it",
        "axes.linewidth": 0.6, "axes.edgecolor": PAL["ink"], "axes.labelcolor": PAL["ink"],
        "axes.labelsize": FS, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": PAL["ink"], "ytick.color": PAL["ink"], "xtick.labelsize": FS - 0.5,
        "ytick.labelsize": FS - 0.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.4,
        "xtick.minor.width": 0.45, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "axes.unicode_minus": True,
    }


MINUS = chr(0x2212)                        # typographic minus -- kept out of the source bytes
MICRO = chr(0x00B5)                        # micro sign, upright (mathtext would italicise mu)


def fmt_num(v, nd=1):
    """Signed number with a true minus sign (U+2212)."""
    s = ("%." + str(nd) + "f") % abs(v)
    return (MINUS + s) if v < 0 else s


# ===========================================================================================
# 4. panels -- draw only
# ===========================================================================================
def _no_frame(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def scalebar(ax, x0, y0, dx=None, dy=None, xlabel=None, ylabel=None, lw=0.8):
    """L-shaped scale bar in DATA units, so its length is exactly what its label says.
    The horizontal bar runs x0 -> x0+dx at y0; the vertical one y0 -> y0+dy at x0+dx."""
    arts = []
    kw = dict(color=PAL["ink"], lw=lw, solid_capstyle="butt", clip_on=False, zorder=6)
    if dx:
        ln, = ax.plot([x0, x0 + dx], [y0, y0], **kw)
        ln.set_gid("scalebar_x:%r" % float(dx))
        arts.append(ln)
        if xlabel:
            arts.append(ax.annotate(xlabel, (x0 + 0.5 * dx, y0), xytext=(0, -2.0),
                                    textcoords="offset points", ha="center", va="top",
                                    fontsize=FS - 0.5, color=PAL["ink"]))
    if dy:
        xv = x0 + (dx or 0.0)
        ln, = ax.plot([xv, xv], [y0, y0 + dy], **kw)
        ln.set_gid("scalebar_y:%r" % float(dy))
        arts.append(ln)
        if ylabel:
            arts.append(ax.annotate(ylabel, (xv, y0 + 0.5 * dy), xytext=(2.2, 0),
                                    textcoords="offset points", ha="left", va="center",
                                    fontsize=FS - 0.5, color=PAL["ink"]))
    return arts


def _nice(x):
    """Largest 1-2-5 step <= x."""
    if not (x > 0):
        return 0.0
    p = 10.0 ** np.floor(np.log10(x))
    for m in (5.0, 2.0, 1.0):
        if m * p <= x * (1.0 + 1e-9):
            return m * p
    return p


def _fmt_bar(v, unit):
    return ("%g %s" % (v, unit)).replace("-", MINUS)


def _mark(ax, x, y, kind, color, size=4.6, z=8):
    """Readout markers, each with a white ring so it stays legible where it sits on a line."""
    if kind == "end":          # open square: DeltaV at t0, and the pinned zero in C
        ax.plot([x], [y], marker="s", ms=size, mfc="white", mec=PAL["ink"], mew=0.9,
                ls="none", zorder=z, clip_on=False)
        return
    mk = {"peak": "o", "bump": "D", "tau": "o", "point": "o"}[kind]
    ms = {"peak": size, "bump": size * 0.92, "tau": size * 0.74, "point": size * 0.6}[kind]
    ax.plot([x], [y], marker=mk, ms=ms, mfc=color, mec="white", mew=0.8, ls="none",
            zorder=z, clip_on=False)


def _label(ax, xy, text, dx=0.0, dy=0.0, ha="center", va="center", size=None, color=None,
           coords="data"):
    return ax.annotate(text, xy, xycoords=coords, xytext=(dx, dy), textcoords="offset points",
                       ha=ha, va=va, fontsize=size or FS + 0.5, color=color or PAL["ink"],
                       zorder=9, annotation_clip=False)


def _band(ax, x, y, color, lw=3.3, alpha=0.38, z=2):
    """A fitted curve drawn as a translucent band, so the data line stays visible on top."""
    ok = np.isfinite(y)
    ax.plot(np.asarray(x)[ok], np.asarray(y)[ok], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", solid_joinstyle="round", zorder=z)


def _guide(ax, xs, ys):
    """Thin dotted construction line (the geometry of a readout, not data)."""
    ax.plot(xs, ys, color=PAL["ink2"], lw=0.5, ls=(0, (1.1, 1.3)), zorder=4)


def _span_arrow(ax, p0, p1, style="|-|,widthA=0.16,widthB=0.16", shrink_end=0.0):
    """A measured interval: a thin bracket (|-|) or double arrow (<|-|>) between two points.
    shrink_end (points) stops the arrow short of p1, where a marker sits."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=style, lw=0.6, color=PAL["ink"], shrinkA=0,
                                shrinkB=shrink_end), zorder=6)


def panel_a(ax, q):
    """Soma Vm: stimulated, sham, the DeltaV between them, the fitted reconstruction.

    Returns the zoom box (the stretch panel B enlarges) in data coordinates."""
    m = q["meta"]
    t, v, ts, vs = q["A_t"], q["A_v"], q["A_ts"], q["A_vs"]
    tm, vm = q["A_tm"], q["A_vm"]
    vrest = float(m["v_rest_mV"])
    x_lo, x_hi = -float(m["baseline_ms"]), float(m["bump_ms"])
    # The view is set by the SLOW part (after the fixed-dt margin) and the rest level. The
    # sub-millisecond transient at t = 0 can be several times larger than the bump; it is
    # clipped here, boxed, and drawn in full in panel B.
    slow = t >= float(m["play_margin_ms"])
    lo = min(float(np.min(v[slow])), float(np.min(vs)), vrest)
    hi = max(float(np.max(v[slow])), float(np.max(vs)), vrest)
    rng = hi - lo
    dy_bar = _nice(0.45 * rng)
    # room under the baseline for the vertical scale bar, clear of the trace's resting end
    y_lo = min(lo - 0.10 * rng, vrest - dy_bar - 0.30 * rng)
    y_hi = hi + 0.16 * rng
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    _no_frame(ax)

    late = t >= float(m["play_margin_ms"])
    vs_on_t = np.interp(t[late], ts, vs)
    ax.fill_between(t[late], vs_on_t, v[late], color=PAL["slow"], alpha=0.13, lw=0, zorder=1)
    ax.plot(ts, vs, color=PAL["sham"], lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax.plot(t, v, color=PAL["ink"], lw=0.85, zorder=3)   # clipped to the view at t = 0
    good = np.isfinite(vm)
    ax.plot(tm[good], vm[good], color=PAL["model"], lw=0.9, ls=(0, (2.4, 1.8)), zorder=4)

    # DeltaV at the bump, measured on the two traces themselves
    if q.get("A_dv_arrow"):
        ta, ya, yb = q["A_dv_arrow"]
        _span_arrow(ax, (ta, ya), (ta, yb), "<|-|>,head_length=0.26,head_width=0.13")
        _label(ax, (ta, 0.5 * (ya + yb)), r"$\Delta V$", dx=3.0, ha="left")

    # resting potential: the one absolute level the scale bars need
    _label(ax, (x_lo, vrest), fmt_num(vrest, 1) + " mV", dx=-3.0, ha="right",
           size=FS - 0.5, color=PAL["ink2"])

    # zoom box around the pulse (enlarged in panel B), wide enough to see at this scale
    zb = max(4.0, 0.0065 * (x_hi - x_lo))
    box = (-zb, y_lo + 0.02 * rng, zb, y_hi - 0.02 * rng)
    ax.add_patch(matplotlib.patches.Rectangle((box[0], box[1]), box[2] - box[0],
                                              box[3] - box[1], fill=False, lw=0.5,
                                              ec=PAL["ink2"], zorder=5))
    scalebar(ax, x_hi - 100.0, y_lo + 0.08 * rng, dx=100.0, dy=dy_bar, xlabel="100 ms",
             ylabel=_fmt_bar(dy_bar, "mV"))
    return dict(box=box, y_top=y_hi)


def draw_pulse_glyph(ax, q):
    """The biphasic current as a small drawing, placed above panel A's t = 0."""
    t, cur = q["B_It"], q["B_I"]
    phi = float(q["meta"]["phase_dur_ms"])
    m = (t >= -2.0 * phi - 0.06) & (t <= 0.06)
    ax.fill_between(t[m], 0, cur[m], color=PAL["wave"], alpha=0.35, lw=0)
    ax.plot(t[m], cur[m], color=PAL["wave"], lw=0.7)
    ax.set_xlim(-2.0 * phi - 0.06, 0.06)
    a = 1.1 * float(np.max(np.abs(cur)))
    ax.set_ylim(-a, a)
    _no_frame(ax)
    ax.patch.set_alpha(0.0)


def panel_b_current(ax, q):
    """The stimulus current over panel B's time range, with its amplitude as a scale bar."""
    phi = float(q["meta"]["phase_dur_ms"])
    ax.axvspan(-2.0 * phi, -phi, color=PAL["pulse"], lw=0, zorder=0)
    ax.axvspan(-phi, 0.0, color=PAL["pulse2"], lw=0, zorder=0)
    ax.fill_between(q["B_It"], 0, q["B_I"], color=PAL["wave"], alpha=0.30, lw=0, zorder=1)
    ax.plot(q["B_It"], q["B_I"], color=PAL["wave"], lw=0.75, zorder=2)
    ax.set_xlim(*q["B_xlim"])
    a = float(np.max(np.abs(q["B_I"])))
    ax.set_ylim(-1.2 * a, 1.2 * a)
    _no_frame(ax)
    x = q["B_xlim"][1] - 0.02 * (q["B_xlim"][1] - q["B_xlim"][0])
    scalebar(ax, x, 0.0, dy=a, ylabel="%g %sA" % (a, MICRO))


def panel_b(ax, q):
    """The direct polarisation: DeltaV_end, the peak, the exponential from it, t_1/e."""
    phi = float(q["meta"]["phase_dur_ms"])
    t, dv = q["B_t"], q["B_dv"]
    t_end, v_end = q["B_end"]
    t_pk, v_pk = q["B_peak"]
    has_pk = bool(np.isfinite(t_pk) and np.isfinite(v_pk))
    if not has_pk:                              # no post-pulse extremum: mark DeltaV_end only
        t_pk, v_pk = t_end, v_end
    s = 1.0 if v_pk >= 0 else -1.0             # the side the fast response lies on
    ax.axvspan(-2.0 * phi, -phi, color=PAL["pulse"], lw=0, zorder=0)
    ax.axvspan(-phi, 0.0, color=PAL["pulse2"], lw=0, zorder=0)
    ax.axhline(0.0, color=PAL["hair"], lw=0.5, zorder=1)
    if q["B_fit_t"] is not None:
        ft, fv = q["B_fit_t"], q["B_fit_v"]
        inw = ft <= q["B_fit_hi"] + 1e-9
        _band(ax, ft[inw], fv[inw], PAL["fast"])                   # the window it was fitted on
        ax.plot(ft[~inw], fv[~inw], color=PAL["fast"], lw=0.7, ls=(0, (1.3, 1.5)),
                alpha=0.85, zorder=2)                              # its continuation
    ax.plot(t, dv, color=PAL["ink"], lw=0.85, zorder=3)
    ax.set_xlim(*q["B_xlim"])
    lo, hi = float(np.min(dv)), float(np.max(dv))
    rng = hi - lo
    beyond = v_pk + s * 0.24 * rng             # room past the peak for its label
    y_lo = min(lo, beyond) - 0.04 * rng
    y_hi = max(hi, beyond) + 0.04 * rng
    ax.set_ylim(y_lo, y_hi)
    _no_frame(ax)

    # t_1/e: peak -> the point where |DeltaV| has fallen to |peak|/e, measured on the data.
    # Drawn on the INNER side (between the curve and zero), where the decay leaves room.
    if has_pk and q["B_t1e"] is not None:
        ta, tb, ye = q["B_t1e"]
        yb = 0.5 * ye
        _guide(ax, [ta, ta], [v_pk, yb])
        _guide(ax, [tb, tb], [ye, yb])
        _bracket(ax, ta, tb, yb, "bracket:t_1e")
        _mark(ax, tb, ye, "point", PAL["fast"])
        _label(ax, (0.5 * (ta + tb), yb), r"$t_{1/e}$", dy=s * 1.6,
               va="bottom" if s > 0 else "top")
    # tau_m: at the end of the fitted window, on the outer side
    if q["B_fit_t"] is not None and np.isfinite(q["B_fit_hi"]):
        ft, fv = q["B_fit_t"], q["B_fit_v"]
        j = int(np.argmin(np.abs(ft - q["B_fit_hi"])))
        _label(ax, (ft[j], fv[j]), r"$\tau_{\mathrm{m}}$", dx=3.0, dy=s * 5.0, ha="left",
               va="bottom" if s > 0 else "top")
    # DeltaV_end and the peak: distinct points, or one point when the peak IS at t0
    same = abs(t_pk - t_end) < 0.5 * float(q["meta"]["dt_ms"])
    if not has_pk:
        _mark(ax, t_end, v_end, "end", None)
        _label(ax, (t_end, v_end), r"$\Delta V_{\mathrm{end}}$", dx=-4.5, ha="right")
    elif same:
        _mark(ax, t_end, v_end, "end", None, size=6.2)
        _mark(ax, t_pk, v_pk, "peak", PAL["fast"], size=3.2)
        _label(ax, (t_end, v_end), r"$\Delta V_{\mathrm{end}} = \Delta V_{\mathrm{pk}}$",
               dx=4.0, dy=s * 4.0, ha="left", va="bottom" if s > 0 else "top")
    else:
        _mark(ax, t_pk, v_pk, "peak", PAL["fast"])
        _mark(ax, t_end, v_end, "end", None)
        _label(ax, (t_end, v_end), r"$\Delta V_{\mathrm{end}}$", dx=-4.5, ha="right")
        _label(ax, (t_pk, v_pk), r"$\Delta V_{\mathrm{pk}}$", dy=s * 4.5,
               va="bottom" if s > 0 else "top")
    _label(ax, (0.0, y_hi), r"$t_0$", dx=2.0, dy=-1.0, ha="left", va="top", size=FS,
           color=PAL["ink2"])
    span = q["B_xlim"][1] - q["B_xlim"][0]
    dx = _nice(0.25 * span)
    dy = _nice(0.30 * rng)
    scalebar(ax, q["B_xlim"][1] - dx - 0.03 * span, y_lo + 0.05 * rng, dx=dx, dy=dy,
             xlabel=_fmt_bar(dx, "ms"), ylabel=_fmt_bar(dy, "mV"))


def _bracket(ax, x0, x1, y, gid):
    """A time readout as a dimension line |-| from x0 to x1 at height y (tagged for tests)."""
    ann = ax.annotate("", xy=(x1, y), xytext=(x0, y),
                      arrowprops=dict(arrowstyle="|-|,widthA=0.16,widthB=0.16", lw=0.6,
                                      color=PAL["ink"], shrinkA=0, shrinkB=0),
                      zorder=6, annotation_clip=False)
    ann.set_gid(gid)
    return ann


def panel_c(ax, q):
    """The bump alone, read out the way B reads out the direct polarisation.

    The data are DeltaV minus the fast term, so they start from 0 at t0: the value the
    trace had there, DeltaV_end (lower square), is taken as the bump's zero (upper square).
    The fit is the pinned double exponential, drawn with its two terms,
    +A_b exp(-u/tau_d) (dashed) and -A_b exp(-u/tau_r) (dotted). As t_1/e in B, each time
    constant is a dimension line from where its term starts (t0) to where it has fallen to
    1/e; t_b runs from t0 to the peak and DeltaV_b rises from the zero to it.
    """
    u, data, fitc, show = q["C_u"], q["C_data"], q["C_fit"], q["C_show"]
    u_hi = float(u[-1])
    amp = float(q["bump"].get("amp_mV", np.nan))
    has_terms = q["C_term_u"] is not None
    _t_end, v_end = q["C_end"]
    tp, vp = q["C_peak"]

    ax.plot([0.0, u_hi], [0.0, 0.0], color=PAL["hair"], lw=0.5, zorder=1)
    if has_terms:
        tu = q["C_term_u"]
        ax.plot(tu, q["C_term_d"], color=PAL["slow"], lw=0.8, ls=(0, (4.0, 2.0)), alpha=0.85,
                zorder=2)
        ax.plot(tu, q["C_term_r"], color=PAL["slow"], lw=0.9, ls=(0, (1.0, 1.5)), alpha=0.9,
                zorder=2)
    _band(ax, u, fitc, PAL["slow"], lw=3.4, alpha=0.40)
    ax.plot(u[show], data[show], color=PAL["ink"], lw=0.85, zorder=3)

    ext = [0.0]
    d = np.asarray(data[show], float)
    d = d[np.isfinite(d)]
    if d.size:
        ext += [float(d.min()), float(d.max())]
    if np.isfinite(v_end):
        ext.append(v_end)
    if has_terms:
        ext += [amp, -amp]
    lo, hi = float(min(ext)), float(max(ext))
    rng = hi - lo
    s_b = 1.0 if (amp >= 0 or not np.isfinite(amp)) else -1.0
    # dimension lines for tau_d and tau_r sit just beyond the terms they measure
    y_d = s_b * (abs(amp) + 0.09 * rng) if has_terms else np.nan
    y_r = -s_b * (abs(amp) + 0.09 * rng) if has_terms else np.nan
    y_lo = (min(lo, y_r) if has_terms else lo) - 0.13 * rng
    y_hi = (max(hi, y_d) if has_terms else hi) + 0.13 * rng
    ax.set_xlim(-0.17 * u_hi, u_hi)
    ax.set_ylim(y_lo, y_hi)
    _no_frame(ax)

    # the pinning: DeltaV_end (the trace at t0) is taken as the bump's zero
    if np.isfinite(v_end):
        _mark(ax, 0.0, v_end, "end", None)
        _label(ax, (0.0, v_end), r"$\Delta V_{\mathrm{end}}$", dx=-4.5, ha="right")
        gap_pt = abs(ax.transData.transform((0.0, v_end))[1]
                     - ax.transData.transform((0.0, 0.0))[1]) * 72.0 / ax.figure.dpi
        if gap_pt > 12.0:
            ax.annotate("", xy=(0.0, 0.0), xytext=(0.0, v_end),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.14",
                                        lw=0.6, color=PAL["ink"], shrinkA=3.4, shrinkB=3.4),
                        zorder=6)
    _mark(ax, 0.0, 0.0, "end", None)

    # tau_d and tau_r: from t0, where each term starts, to its 1/e point
    if has_terms:
        for key, y_dim, lab, gid in (("C_tau_d", y_d, r"$\tau_{\mathrm{d}}$", "bracket:tau_d"),
                                     ("C_tau_r", y_r, r"$\tau_{\mathrm{r}}$", "bracket:tau_r")):
            tx, ty = q[key]
            # the line's left tick marks t0 itself; only the 1/e end needs a guide
            _guide(ax, [tx, tx], [ty, y_dim])
            _bracket(ax, 0.0, tx, y_dim, gid)
            _mark(ax, tx, ty, "tau", PAL["slow"])
            sd = 1.0 if y_dim > 0 else -1.0
            _label(ax, (0.5 * tx, y_dim), lab, dy=sd * 1.8, va="bottom" if sd > 0 else "top")

    # the peak: t_b from t0 (dimension line between the bump and the + term), DeltaV_b from 0
    if np.isfinite(tp) and np.isfinite(vp):
        s = 1.0 if vp >= 0 else -1.0
        if has_terms:
            env = amp * np.exp(-tp / float(q["C_tau_d"][0]))       # the + term at t_b
            y_tb = vp + 0.45 * (env - vp)
        else:
            y_tb = vp + s * 0.10 * rng
        _guide(ax, [tp, tp], [vp, y_tb])
        _bracket(ax, 0.0, tp, y_tb, "bracket:t_b")
        # the label rides on the bracket; slide it toward t0 until it clears the + term
        xl = 0.5 * tp
        if has_terms:
            need = abs(ax.transData.inverted().transform((0, 12.0 * ax.figure.dpi / 72.0))[1]
                       - ax.transData.inverted().transform((0, 0.0))[1])
            while xl > 0.2 * tp and s * (amp * np.exp(-xl / float(q["C_tau_d"][0]))
                                         - y_tb) < need:
                xl *= 0.85
        _label(ax, (xl, y_tb), r"$t_{\mathrm{b}}$", dy=s * 1.6,
               va="bottom" if s > 0 else "top")
        _span_arrow(ax, (tp, 0.0), (tp, vp), "<|-|>,head_length=0.26,head_width=0.13",
                    shrink_end=3.2)
        _mark(ax, tp, vp, "bump", PAL["slow"])
        _label(ax, (tp, 0.45 * vp), r"$\Delta V_{\mathrm{b}}$", dx=2.5, ha="left")

    dx = _nice(0.25 * u_hi)
    dy = _nice(0.22 * rng)
    # inset from the right edge so the bar's label stays on the page
    scalebar(ax, u_hi - dx - 0.12 * u_hi, y_lo + 0.05 * rng, dx=dx, dy=dy,
             xlabel=_fmt_bar(dx, "ms"), ylabel=_fmt_bar(dy, "mV"))


# ===========================================================================================
# 5. layout -- millimetre geometry
# ===========================================================================================
def _ax_mm(fig, x, y, w, h, W, H):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def draw_key(fig, W, H, y_mm, x0_mm, x1_mm):
    """One legend row for the whole figure, each encoding once, spaced by measured width."""
    ax = _ax_mm(fig, x0_mm, y_mm - 2.5, x1_mm - x0_mm, 5.0, W, H)
    ax.set_xlim(0, x1_mm - x0_mm)
    ax.set_ylim(-1, 1)
    _no_frame(ax)
    ax.patch.set_alpha(0.0)
    items = [("line", dict(color=PAL["ink"], lw=0.85), "stimulated"),
             ("line", dict(color=PAL["sham"], lw=0.8, ls=(0, (3.0, 2.0))), "sham"),
             ("band", PAL["fast"], "direct polarization"),
             ("band", PAL["slow"], r"$I_{\mathrm{h}}$ bump"),
             ("line", dict(color=PAL["model"], lw=0.9, ls=(0, (2.4, 1.8))), "fitted sum")]
    renderer = fig.canvas.get_renderer()
    sw, gap_txt, gap_item = 6.0, 1.3, 5.5                        # mm
    x = 0.0
    for kind, style, text in items:
        if kind == "line":
            ax.plot([x, x + sw], [0, 0], **style)
        else:
            ax.plot([x + 0.6, x + sw - 0.6], [0, 0], color=style, lw=3.3, alpha=0.45,
                    solid_capstyle="round")
        tx = ax.text(x + sw + gap_txt, -0.36, text, ha="left", va="baseline", fontsize=FS,
                     color=PAL["ink"])
        w_mm = tx.get_window_extent(renderer=renderer).width / fig.dpi * 25.4
        x += sw + gap_txt + w_mm + gap_item
    # centre the whole row
    total = x - gap_item
    shift = 0.5 * ((x1_mm - x0_mm) - total)
    for ln in ax.lines:
        ln.set_xdata(np.asarray(ln.get_xdata(), float) + shift)
    for tx in ax.texts:
        px, py = tx.get_position()
        tx.set_position((px + shift, py))
    return ax


def make_figure(q, width_mm=180.0, height_mm=118.0):
    """Assemble the three panels at fixed millimetre positions. Returns (fig, axes dict).

    Horizontal positions scale with width_mm; the vertical layout is fixed for 118 mm and
    scales proportionally otherwise."""
    W, H = float(width_mm), float(height_mm)
    sx, sy = W / 180.0, H / 118.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = {}
    ax["A"] = _ax_mm(fig, 21 * sx, 64 * sy, 149 * sx, 40 * sy, W, H)
    ax["B_I"] = _ax_mm(fig, 9 * sx, 47 * sy, 74 * sx, 7 * sy, W, H)
    ax["B"] = _ax_mm(fig, 9 * sx, 8 * sy, 74 * sx, 37.5 * sy, W, H)
    ax["C"] = _ax_mm(fig, 101 * sx, 8 * sy, 76 * sx, 46 * sy, W, H)

    geo = panel_a(ax["A"], q)
    # the pulse drawing above t = 0 of panel A
    to_fig = fig.transFigure.inverted()
    xf = to_fig.transform(ax["A"].transData.transform((0.0, 0.0)))[0]
    yf = to_fig.transform(ax["A"].transData.transform((0.0, geo["y_top"])))[1]
    gw, gh = 8.0 / W, 4.0 / H
    ax["A_glyph"] = fig.add_axes([xf - 0.5 * gw, yf + 0.4 / H, gw, gh])
    draw_pulse_glyph(ax["A_glyph"], q)

    panel_b_current(ax["B_I"], q)
    panel_b(ax["B"], q)
    panel_c(ax["C"], q)

    # zoom lines: the box around the pulse in A -> the top corners of B
    bx0, by0, bx1, _ = geo["box"]
    for (xa, xb) in ((bx0, 0.0), (bx1, 1.0)):
        con = matplotlib.patches.ConnectionPatch(
            xyA=(xa, by0), coordsA=ax["A"].transData, xyB=(xb, 1.0), coordsB=ax["B_I"].transAxes,
            color=PAL["ink2"], lw=0.45, zorder=0)
        fig.add_artist(con)

    draw_key(fig, W, H, y_mm=H - 4.5 * sy, x0_mm=21 * sx, x1_mm=170 * sx)
    for letter, (x, y) in dict(A=(2.5, 105.0), B=(2.5, 55.5), C=(91.0, 55.5)).items():
        fig.text(x * sx / W, y * sy / H, letter, fontsize=FS + 2.0, fontweight="bold",
                 color=PAL["ink"], ha="left", va="bottom")
    return fig, ax


# ===========================================================================================
# 6. export
# ===========================================================================================
def save_figure(fig, stem, formats=("pdf", "svg", "png"), dpi=600):
    d = os.path.dirname(os.path.abspath(stem))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    out = []
    for fmt in formats:
        p = "%s.%s" % (stem, fmt)
        fig.savefig(p, dpi=dpi, facecolor="white")
        out.append(p)
    return out


def values_text(q):
    """Every number read out of this trace, with the campaign CSV's column names."""
    m, v = q["meta"], q["values"]
    lines = ["# paper_fig_readouts -- %s L%d um, soma (%g, %g) um, theta %g deg, %s"
             % (m["morph"], int(m["layer_um"]), m["x_um"], m["y_um"], m["theta_deg"],
                m["cell_model"]),
             "# +/-%g uA, %g ms per phase, ramp %g us; window %g ms; baseline %g ms"
             % (m["i0_uA"], m["phase_dur_ms"], m["ramp_us"], m["bump_ms"], m["baseline_ms"])]
    for k in sorted(v):
        val = v[k]
        lines.append("%-24s %s" % (k, ("%.6g" % val) if isinstance(val, float) else val))
    return "\n".join(lines) + "\n"


# ===========================================================================================
# 7. checks -- the drawing is data-dependent, so collisions are checked, not assumed
# ===========================================================================================
def _drawn_tick_labels(ax):
    """(all tick-label Text objects of ax, the subset inside the view interval). Tick labels
    for locations outside the view exist as artists but are never drawn."""
    all_, drawn = [], []
    for axis, idx in ((ax.xaxis, 0), (ax.yaxis, 1)):
        lo, hi = sorted(axis.get_view_interval())
        tol = 1e-9 * max(1.0, abs(hi - lo))
        for which in (False, True):
            for lab in axis.get_ticklabels(minor=which):
                all_.append(lab)
                pos = lab.get_position()[idx]
                if lo - tol <= pos <= hi + tol:
                    drawn.append(lab)
    return all_, drawn


def text_boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tick_all, tick_drawn = set(), set()
    for ax in fig.axes:
        a, d = _drawn_tick_labels(ax)
        tick_all.update(map(id, a))
        tick_drawn.update(map(id, d))
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in tick_all and id(t) not in tick_drawn:
            continue
        bb = t.get_window_extent(renderer=r)
        if bb.width > 0 and bb.height > 0:
            out.append((t.get_text(), bb))
    return out


def layout_problems(fig, shrink_px=0.5):
    """(overlapping text pairs, text outside the page). Empty lists = clean."""
    boxes = text_boxes(fig)
    over = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > shrink_px and
                    min(a.y1, b.y1) - max(a.y0, b.y0) > shrink_px):
                over.append((boxes[i][0], boxes[j][0]))
    W, H = fig.bbox.width, fig.bbox.height
    outside = [s for s, bb in boxes if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5
               or bb.y1 > H + 0.5]
    return over, outside


# ===========================================================================================
# driver
# ===========================================================================================
def render(q, stem, width_mm=180.0, height_mm=118.0, formats=("pdf", "svg", "png"),
           font=None, verbose=True):
    with plt.rc_context(paper_rc(font)):
        fig, _ = make_figure(q, width_mm, height_mm)
        over, outside = layout_problems(fig)
        paths = save_figure(fig, stem, formats)
        plt.close(fig)
    with open(stem + "_values.txt", "w") as fh:
        fh.write(values_text(q))
    if verbose:
        for p in paths + [stem + "_values.txt"]:
            print("wrote", p)
        if over:
            print("WARNING: overlapping labels:", over)
        if outside:
            print("WARNING: text outside the page:", outside)
    return paths, over, outside


def _cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "all"):
        p = sub.add_parser(name)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--placement", help="morph:layer:x:y:theta")
        g.add_argument("--from-examples-csv", help="plot_vm_examples.csv / campaign_examples.csv")
        p.add_argument("--example", type=int, default=1, help="with --from-examples-csv")
        p.add_argument("--cell-model", default="full_tuned")
        p.add_argument("--bump-ms", type=float, default=800.0)
        p.add_argument("--baseline-ms", type=float, default=100.0)
        p.add_argument("--out-stem", default="fig_readouts")
    p = sub.add_parser("suggest", help="rank the neurons of an examples CSV for the figure")
    p.add_argument("--from-examples-csv", required=True)
    p.add_argument("--top", type=int, default=8)
    p = sub.add_parser("plot")
    p.add_argument("--data", required=True)
    p.add_argument("--out-stem", default=None)
    for p in (sub.choices["all"], sub.choices["plot"]):
        p.add_argument("--width-mm", type=float, default=180.0)
        p.add_argument("--height-mm", type=float, default=118.0)
        p.add_argument("--font", default=None, help="font family (default: Arial, then "
                       "Helvetica / Liberation Sans / ...)")
    a = ap.parse_args(argv)

    if a.cmd == "suggest":
        ranked = suggest_examples(a.from_examples_csv, a.top)
        if not ranked:
            print("no row has both fits accepted -- nothing to suggest")
            return 1
        print("%-7s %-7s %-7s %5s  %-15s %8s %8s %8s %8s %7s %7s"
              % ("example", "morph", "layer", "d_um", "x,y (um)", "dV_end", "dV_pk",
                 "t_pk", "bump", "t_b", "r2"))
        for sc, ex, r in ranked:
            print("%-7d %-7s %-7s %5.0f  %-15s %8.3f %8.3f %8.3f %8.3f %7.1f %7.4f"
                  % (ex, r.get("morph", "?"), r.get("layer_um", "?"), _f(r, "r_um"),
                     "%.0f,%.0f" % (_f(r, "x_um"), _f(r, "y_um")),
                     _f(r, "dexp_dv_t0_mV"), _f(r, "early_peak_mV"),
                     _f(r, "early_t_peak_ms"), _f(r, "dexp_peak_mV"),
                     _f(r, "dexp_t_peak_ms"), _f(r, "dexp_r2")))
        print("then:  python paper_fig_readouts.py all --from-examples-csv %s --example N"
              % a.from_examples_csv)
        return 0

    if a.cmd in ("simulate", "all"):
        pl = (parse_placement(a.placement) if a.placement
              else placement_from_examples_csv(a.from_examples_csv, a.example))
        print("simulating %(morph)s L%(layer)g at (%(x_um)g, %(y_um)g) um, theta %(theta_deg)g"
              % pl, flush=True)
        rec, meta = acquire(pl, cell_model=a.cell_model, bump_ms=a.bump_ms,
                            baseline_ms=a.baseline_ms)
        data = save_record(a.out_stem + ".npz", rec, meta)
        print("wrote", data)
        if a.cmd == "simulate":
            return 0
        stem = a.out_stem
    else:
        rec, meta = load_record(a.data)
        stem = a.out_stem or os.path.splitext(a.data)[0]
    q = readouts(rec, meta)
    render(q, stem, a.width_mm, a.height_mm, font=a.font)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
