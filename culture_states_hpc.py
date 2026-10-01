#!/usr/bin/env python3
"""
culture_states_hpc.py -- three-outcome snapshot of ONE simulated culture, from the campaign table.

SELF-CONTAINED: one file; needs only numpy, pandas, matplotlib (+ morphio for the grey
morphologies). No NEURON, no other script of the project. See HOWTO_culture_states.md.

What it draws (3 panels, one culture, one slab thickness)
    1  neurons that FIRED at any time of the simulated window        -> green somata
    2  neurons DEPOLARIZED at t0  (end of phase 2)                   -> red gradient (by dVm)
    3  neurons HYPERPOLARIZED at t0                                   -> blue gradient (by dVm)
  Other somata = small light-grey dots; electrodes = outlines; morphologies (optional) = thin grey.
  Rules (same as culture_statistics.polarization_labels): a spike wins; otherwise
  dVm >= +neu -> depolarized, dVm <= -neu -> hyperpolarized (default neu = 1 mV), else neutral.
  dVm = `deltaVm_end_phase2_mV` of the merged table (stim - sham at t0). The CSV outcome labels
  (sign only, no threshold) are NOT used, except `activation` (= a spike).

Input (--input): ONE table of the campaign -- a merged_<outcome>.csv of culture_statistics, a
culture_P<outcome>.csv (culture_export, or a per-job / all-jobs merge of culture_merge), or one raw
worker part (results_<model>/parts_<job>/part_NNN.csv). All carry the same neurons and outcomes: one
file is enough, NO merge needed. It can be huge (millions of rows): it is read in chunks and filtered
on the fly. Cultures are named <model>_S<seed>_C<c> in merged_*.csv files and S<seed>_C<c> in the
others (F<file>_C<c> / C<c> without a seed column), c = the culture index the campaign drew from:
`local_culture` when the file has it (culture_merge renumbers `culture` to a global id 0..K-1 and
keeps the per-job index there), else `culture` (see --list).

Grey morphologies (optional): the table only has theta_orient_deg (folded to 0-90), so the full
rotation of each neuron is REBUILT from the campaign draws (rng = default_rng(seed + c): shuffle of
the morphology indices, positions, rotations -- as culture_export.culture_draws), and CHECKED against
x_um / y_um and theta_orient_deg of the table. If it does not match, nothing is drawn and the script
says so. Seed, neurons per culture, number of specimens and the soma square -- half side S and centre
(cx, cy): +/-500 um around (0, 0) in the campaigns before 2026-09-28, +/-300 um around the dipole centre
(0, -30) since -- are INFERRED from the table (override with --seed --n-neurons --n-morph --span
--center); a culture still being written (a running job's part) is fine: its size is searched and
checked on the rotations. With a file that has no `seed` column, give --seed (or use --find-seed).
The sliced dendrites (nested slab, >= 80% of the arc length inside, as slicer.py) are drawn in the
plane (x, y); the stylised axon is not drawn.

Typical use
    python culture_states_hpc.py --selftest
    python culture_states_hpc.py --input stats/activation/merged_activation.csv --list
    python culture_states_hpc.py --input stats/activation/merged_activation.csv --layer 80 \
        --culture full_tuned_S50000_C0 --morph-dir eyal_archive --require-all-morph --out figures/culture_states
  In the repository, as a batch job:  qsub -v INPUT=...,CULTURE=... jobs/culture_states.pbs
"""
import argparse
import contextlib
import glob
import io
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

NEED = ["culture_global", "neuron", "layer_um", "morphology", "x_um", "y_um",
        "deltaVm_end_phase2_mV", "phase2_outcome"]
GREEN = (0.12, 0.62, 0.24)
GRAY_LEVELS = ["#3d3d3d", "#808080", "#b8b8b8", "#5e5e5e", "#9e9e9e", "#c8c8c8"]
ELEC_UM = 25.0


# ----------------------------------------------------------------------------- reading
BASE = ["neuron", "layer_um", "morphology", "x_um", "y_um", "deltaVm_end_phase2_mV", "phase2_outcome"]


def _add_culture_name(ch, path=""):
    """Ensure a `culture_global` column: merged_*.csv files have it; the others get S<seed>_C<c>, with c
    the culture index the campaign drew from (rng = default_rng(seed + c)): `local_culture` when present
    -- culture_merge renumbers `culture` to a global id 0..K-1 and keeps the per-job index there --
    else `culture` (raw worker parts, culture_export's own files). Rows that cannot be named or placed
    (a blank culture name / index / seed, neuron or layer) are dropped with a NOTE."""
    if "culture_global" in ch.columns:
        cidx, key = None, [ch["culture_global"]]
    else:
        if "local_culture" not in ch.columns and "culture" not in ch.columns:
            raise SystemExit("neither `culture_global` nor `culture` in the file")
        cidx = ch["local_culture"] if "local_culture" in ch.columns else ch["culture"]
        if "local_culture" in ch.columns and "culture" in ch.columns:
            cidx = cidx.fillna(ch["culture"])
        key = [cidx] + ([ch["seed"]] if "seed" in ch.columns else [])
    key += [ch[c] for c in ("neuron", "layer_um") if c in ch.columns]
    bad = np.zeros(len(ch), bool)
    for col in key:
        bad |= col.isna().to_numpy()
    if bad.any():
        print(f"  NOTE: {path}: {int(bad.sum())} row(s) with a blank identity column dropped")
        ch = ch[~bad]
        cidx = None if cidx is None else cidx[~bad]
    if cidx is not None:
        ch = ch.copy()
        if "seed" in ch.columns:
            ch["culture_global"] = "S" + ch["seed"].astype(int).astype(str) + "_C" + cidx.astype(int).astype(str)
        else:
            ch["culture_global"] = "C" + cidx.astype(int).astype(str)
    return ch


class _Head(io.RawIOBase):
    """The first `size` bytes of a file: a fixed snapshot, whatever a job still appending to it writes."""

    def __init__(self, path, size):
        super().__init__()
        self._fh, self._left = open(path, "rb"), int(size)

    def readable(self):
        return True

    def readinto(self, b):
        data = self._fh.read(min(len(b), self._left)) if self._left > 0 else b""
        b[:len(data)] = data
        self._left -= len(data)
        return len(data)

    def close(self):
        self._fh.close()
        super().close()


def _csv_chunks(path, keep, chunksize):
    """pd.read_csv chunks of the columns in `keep`, from a snapshot of the file's current size. A raw part
    of a job that is still running can end with a half-written row (no final newline): that row is dropped,
    as culture_merge does."""
    size = os.path.getsize(path)
    if size == 0:
        raise SystemExit(f"{path} is empty")
    with open(path, "rb") as fh:
        fh.seek(size - 1)
        complete = fh.read(1) == b"\n"
    prev = None
    with io.BufferedReader(_Head(path, size)) as src:
        for ch in pd.read_csv(src, usecols=lambda c: c in keep, chunksize=chunksize):
            if prev is not None:
                yield prev
            prev = ch
    if prev is not None:
        if not complete and len(prev):
            print(f"  NOTE: {path}: the last row is incomplete (file still being written?) -- dropped")
            prev = prev.iloc[:-1]
        yield prev


def _read_filtered(path, layer, culture, one_culture, chunksize):
    """Chunked read keeping the needed columns and rows (requested layer; the requested culture,
    or the first culture of the file if one_culture, else all)."""
    keep = set(BASE) | {"culture_global", "culture", "local_culture", "seed", "fired", "theta_orient_deg"}
    parts, first = [], culture
    for ch in _csv_chunks(path, keep, chunksize):
        miss = [c for c in BASE if c not in ch.columns]
        if miss:
            raise SystemExit(f"{path}: missing columns {miss} (is it a raw culture_P*.csv or a merged_*.csv?)")
        ch = _add_culture_name(ch, path)
        if layer is not None:
            ch = ch[np.isclose(ch["layer_um"].astype(float), float(layer))]
        if first is None and one_culture and len(ch):
            first = ch["culture_global"].iloc[0]
        if first is not None:
            ch = ch[ch["culture_global"] == first]
        parts.append(ch)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BASE)


def list_cultures(path, chunksize=500_000):
    """culture name x layer -> number of neurons, scanning the file in chunks."""
    acc = {}
    keep = {"culture_global", "culture", "local_culture", "seed", "layer_um"}
    for ch in _csv_chunks(path, keep, chunksize):
        ch = _add_culture_name(ch, path)
        for k, v in ch.groupby(["culture_global", "layer_um"]).size().items():
            acc[k] = acc.get(k, 0) + int(v)
    return acc


# ----------------------------------------------------------------------------- outcomes
def states(dv_t0, fired_any, neu):
    """Mutually exclusive outcomes: spike wins; dVm >= +neu depolarized; dVm <= -neu hyperpolarized."""
    f = np.asarray(fired_any) > 0
    dv = np.asarray(dv_t0, float)
    return f, (~f) & (dv >= neu), (~f) & (dv <= -neu)


# ----------------------------------------------------------------------------- rotations
def assign_morphologies(n, n_morph, rng):
    """Copy of culture_export.assign_morphologies: round-robin then shuffle."""
    idx = np.array([i % n_morph for i in range(n)], int)
    rng.shuffle(idx)
    return idx


def draws(seed, c, n_neurons, n_morph, span, center=(0.0, 0.0)):
    """The campaign draws of culture c: (morphology indices, positions [um], rotations [deg]).
    Somata uniform in the square of half side `span` around `center`; the centre is added AFTER the
    draw, so the random stream does not depend on it (as culture_export.culture_draws)."""
    rng = np.random.default_rng(int(seed) + int(c))
    midx = assign_morphologies(int(n_neurons), int(n_morph), rng)
    pos = rng.uniform(-span, span, size=(int(n_neurons), 2)) + np.asarray(center, float).reshape(1, 2)
    theta = rng.uniform(0, 360, size=int(n_neurons))
    return midx, pos, theta


# theta_orient_deg of the tables = the rotation folded to [0, 90] against the dipole axis of the array
# (culture_export.rel_orientation_deg / dipole_axis_deg: anode column -> cathode column = 0 deg for the
# default 3 + 3 array drawn by electrodes() below), rounded to 0.1 deg.
ORIENT_AXIS_DEG = 0.0
ORIENT_TOL_DEG = 0.06


def fold_orientation(theta_deg, axis_deg=ORIENT_AXIS_DEG):
    """Copy of culture_export.rel_orientation_deg: |theta - axis| folded to [0, 90] (a neurite and its
    180-degree flip couple to the field the same way)."""
    d = np.abs((np.asarray(theta_deg, float) - axis_deg) % 180.0)
    return np.minimum(d, 180.0 - d)


def _orient_ok(theta, rows):
    """True if the rebuilt rotations give the table's theta_orient_deg on every row (or if the table has
    no such column). Positions alone are not enough: a culture cut short (a job still running, or killed)
    can leave the random stream where the full culture had it after the shuffle, so EVERY position matches
    while every rotation -- drawn after the 2N position values -- is wrong."""
    if "theta_orient_deg" not in rows.columns:
        return True
    want = rows["theta_orient_deg"].to_numpy(float)
    return bool(np.all(np.abs(fold_orientation(theta) - want) <= ORIENT_TOL_DEG))


class OrientationMismatch(ValueError):
    """The positions are reproduced but the rotations do not give the table's theta_orient_deg."""


NO_ORIENT_HINT = ("if theta_orient_deg of this file follows another definition (an export older than "
                  "culture_export.rel_orientation_deg), --no-orient-check checks the positions only -- "
                  "wrong rotations for a culture cut short")


def rotations_from_seed(d, seed, n_neurons, n_morph, span, center=(0.0, 0.0), tol=0.011, check_orient=True):
    """Full rotation of every row of d, rebuilt from the draws; ValueError if the rebuilt positions
    differ from x_um / y_um, OrientationMismatch if the rebuilt rotations differ from theta_orient_deg
    (check_orient False: positions only). seed=None -> use the `seed` column of the table (raw files)."""
    theta = np.full(len(d), np.nan)
    for cg, idx in d.groupby("culture_global").groups.items():
        c = int(str(cg).split("_C")[-1])
        rows = d.loc[idx]
        if seed is None:
            if "seed" not in rows.columns:
                raise ValueError("no seed given and no `seed` column in the file (use --seed or --find-seed)")
            sd = int(rows["seed"].iloc[0])
        else:
            sd = int(seed)
        _m, pos, th = draws(sd, c, n_neurons, n_morph, span, center)
        n = rows["neuron"].to_numpy(int)
        if n.max() >= n_neurons or not np.allclose(rows[["x_um", "y_um"]].to_numpy(), pos[n], atol=tol):
            raise ValueError(f"positions of {cg} do not match seed={sd}, span={span}, "
                             f"centre=({center[0]:g}, {center[1]:g}), N={n_neurons}")
        if check_orient and not _orient_ok(th[n], rows):
            raise OrientationMismatch(f"rotations of {cg} do not match theta_orient_deg (seed={sd}, "
                                      f"N={n_neurons}): wrong number of neurons per culture? ({NO_ORIENT_HINT})")
        theta[np.asarray(idx)] = th[n]
    return theta


def _fit_square(z, xy, span=None):
    """Least-squares (S, cx, cy) of xy = (cx, cy) + S z, z = 2u - 1 the unit draws; S fixed when given."""
    z, xy = np.asarray(z, float), np.asarray(xy, float)
    if span is not None:
        cen = (xy - float(span) * z).mean(axis=0)
        return float(span), float(cen[0]), float(cen[1])
    k = len(z)
    A = np.zeros((2 * k, 3))
    A[:k, 0], A[k:, 0], A[:k, 1], A[k:, 2] = z[:, 0], z[:, 1], 1.0, 1.0
    sol = np.linalg.lstsq(A, np.concatenate([xy[:, 0], xy[:, 1]]), rcond=None)[0]
    return float(sol[0]), float(sol[1]), float(sol[2])


def _snap(v, nd):
    """v rounded to integers (nd 0), halves ('half'), nd decimals, or left as is (None); never -0."""
    if nd is None:
        return float(v) + 0.0
    if nd == "half":
        return round(2.0 * v) / 2.0 + 0.0
    return round(float(v), nd) + 0.0


N_SCAN = 10000      # culture sizes tried above the highest neuron index present (a culture being written)


def _square_for(z, xy, span, tol):
    """(S, (cx, cy)) reproducing xy = centre + S z within tol, S and centre by least squares then the
    simplest rounding that still fits; None if nothing fits."""
    k = min(len(z), 16)                                    # cheap rejection first (a wrong culture size)
    s0, cx0, cy0 = _fit_square(z[:k], xy[:k], span)
    if not np.allclose(xy[:k], np.array([cx0, cy0]) + z[:k] * s0, atol=tol):
        return None
    s_fit, cx, cy = _fit_square(z, xy, span)
    for nd in (0, "half", 1, 2, None):
        S = float(span) if span is not None else _snap(s_fit, nd)
        cen = (_snap(cx, nd), _snap(cy, nd))
        if S > 0 and np.allclose(xy, np.asarray(cen) + z * S, atol=tol):
            return S, cen
    return None


def infer_params(d, seed=None, n_neurons=None, span=None, tol=0.011, check_orient=True):
    """(n_neurons, n_morph, span, center) from the table of ONE culture; n_morph = number of distinct
    morphologies. The soma square: x = cx + (2u - 1) S, y = cy + (2v - 1) S with (u, v) the campaign draws
    -- centre (0, 0) in the campaigns before 2026-09-28, the dipole centre of the array (0, -30 um) since
    (culture_export.placement_frame) -- S (unless given) and (cx, cy) by least squares, then the simplest
    rounding that still reproduces EVERY position within tol. The draws depend on the culture size N, so
    (unless n_neurons is given) N is searched from the highest neuron index + 1 upwards (by N_SCAN at
    most: a culture still being written has fewer neurons than N) and accepted only if the positions AND
    the rotations (theta_orient_deg) of every row are reproduced (check_orient False: the positions only,
    N = highest index + 1 -- wrong for a culture cut short). ValueError if no N does (wrong seed or file)."""
    cg = sorted(d["culture_global"].unique())[0]
    rows = d[d["culture_global"] == cg].sort_values("neuron")
    c = int(str(cg).split("_C")[-1])
    sd = int(rows["seed"].iloc[0]) if seed is None else int(seed)
    n_morph = int(rows["morphology"].nunique())
    n = rows["neuron"].to_numpy(int)
    xy = rows[["x_um", "y_um"]].to_numpy(float)
    lo = int(n.max()) + 1
    if len(n) < 3 or (n_neurons is not None and lo > int(n_neurons)):
        raise ValueError(f"cannot infer the soma square of {cg} from {len(n)} neuron(s)"
                         + (f" with N={n_neurons}" if n_neurons is not None else "") + ": give --span and --center")
    if n_neurons is not None:
        sizes = [int(n_neurons)]
    else:
        sizes = range(lo, lo + (N_SCAN if check_orient else 0) + 1)
    positions_only = False
    for N in sizes:
        rng = np.random.default_rng(sd + c)
        assign_morphologies(N, n_morph, rng)
        z = 2 * rng.uniform(0, 1, size=(N, 2))[n] - 1
        fit = _square_for(z, xy, span, tol)
        if fit is None:
            continue
        if check_orient and not _orient_ok(rng.uniform(0, 360, size=N)[n], rows):
            positions_only = True
            continue
        return N, n_morph, fit[0], fit[1]
    tried = f"N={sizes[0]}" if len(sizes) == 1 else f"N={sizes[0]}..{sizes[-1]}"
    if positions_only:
        raise OrientationMismatch(f"could not infer the culture size of {cg} ({tried}): the positions are "
                                  f"reproduced but never the rotations ({NO_ORIENT_HINT})")
    raise ValueError(f"could not infer the soma square ({tried}: no square reproduces the positions of {cg}); "
                     f"check that --seed is the campaign seed and the file is a table of the campaign")


def find_seed(d, n_neurons, n_morph, span, lo=0, hi=100000, tol=0.011):
    """Seeds in [lo, hi) whose draws reproduce the positions of the first culture of d. Compares the
    offsets between somata, in which the centre of the square cancels: no centre needed."""
    cg = sorted(d["culture_global"].unique())[0]
    c = int(str(cg).split("_C")[-1])
    rows = d[d["culture_global"] == cg].sort_values("neuron")
    n = rows["neuron"].to_numpy(int)
    xy = rows[["x_um", "y_um"]].to_numpy(float)
    rel = xy - xy[0]
    hits = []
    for seed in range(lo, hi):
        rng = np.random.default_rng(seed + c)
        assign_morphologies(int(n_neurons), int(n_morph), rng)
        pos = rng.uniform(-span, span, size=(int(n_neurons), 2))
        if np.allclose(rel, pos[n] - pos[n[0]], atol=2 * tol):
            hits.append(seed)
    return hits


# ----------------------------------------------------------------------------- morphologies
def _read_tree(asc):
    """soma centre (3,) and {id: dict(type, pts, parent, children)} (as slicer._read_tree)."""
    import morphio
    morphio.set_maximum_warnings(0)
    m = morphio.Morphology(asc)
    sc = np.asarray(m.soma.points, float).mean(0)
    tree = {}
    for s in m.iter():
        tree[s.id] = dict(type=s.type, pts=np.asarray(s.points, float),
                          parent=(None if s.is_root else s.parent.id),
                          children=[c.id for c in s.children])
    return sc, tree


def _frac_in_slab(pts, zc, T):
    """Fraction of a section's arc length inside [zc - T/2, zc + T/2] (as slicer._frac_in_slab)."""
    z = pts[:, 2]
    if len(z) < 2:
        return 1.0 if abs(z[0] - zc) <= T / 2 else 0.0
    lo, hi = zc - T / 2, zc + T / 2
    L = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = 0.0
    for i in range(len(z) - 1):
        a, b = sorted((z[i], z[i + 1]))
        ov = max(0.0, min(b, hi) - max(a, lo))
        f = ov / (b - a) if b > a else (1.0 if lo <= a <= hi else 0.0)
        s += f * L[i]
    return s / L.sum() if L.sum() > 0 else 0.0


def kept_ids(tree, zc, layer, thr=0.80, axon_type=None):
    """Connected subtree of non-axon sections with >= thr of the arc length in the slab."""
    import morphio
    axon = morphio.SectionType.axon if axon_type is None else axon_type
    kept = set()
    stack = [i for i, s in tree.items() if s["parent"] is None]
    while stack:
        i = stack.pop()
        s = tree[i]
        if s["type"] == axon:
            continue
        if (s["parent"] is None or s["parent"] in kept) and _frac_in_slab(s["pts"], zc, layer) >= thr:
            kept.add(i)
            stack.extend(s["children"])
    return kept


def sliced_polylines(asc, layer, thr=0.80):
    """[(n,2) soma-relative polylines] of the kept (dendritic) sections at slab thickness `layer`."""
    sc, tree = _read_tree(asc)
    kept = kept_ids(tree, sc[2], layer, thr)
    return [tree[i]["pts"][:, :2] - sc[:2] for i in kept if len(tree[i]["pts"]) >= 2]


def find_asc(spec, morph_dir):
    """The morphology.asc of specimen `spec`. As the campaign picks it (morphologies.find_one_morphology):
    the first specimen_* folder directly under morph_dir, in sorted order, whose name contains spec;
    otherwise the first match searched recursively (a bundle root, or a folder of .asc files)."""
    root = os.path.expanduser(morph_dir)
    for p in sorted(glob.glob(os.path.join(root, "specimen_*", "morphology.asc"))):
        if spec in os.path.basename(os.path.dirname(p)):
            return p
    for pat in (f"**/*{spec}*/morphology.asc", f"**/*{spec}*.asc"):
        hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
        if hits:
            return hits[0]
    return None


def rotate_translate(pts, x, y, theta_deg):
    """Rotate soma-relative points by theta (counter-clockwise) and move them to (x, y)."""
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.asarray(pts, float) @ R.T + np.array([x, y], float)


# ----------------------------------------------------------------------------- figure
def electrodes(pitch=60.0):
    h = pitch / 2
    ys = np.array([h, -h, -3 * h])
    return np.vstack([np.column_stack([np.full(3, -h), ys]), np.column_stack([np.full(3, h), ys])])


def _trunc(name, lo, hi):
    return LinearSegmentedColormap.from_list(name + "_t", plt.get_cmap(name)(np.linspace(lo, hi, 256)))


def draw_figure(d, theta, polys, layer, half, neu, dvm_max, soma_size, title, out_stem, pitch=60.0,
                morph_alpha=None):
    """Three panels; returns (files, (n_act, n_dep, n_hyp))."""
    xy = d[["x_um", "y_um"]].to_numpy(float)
    dv = d["deltaVm_end_phase2_mV"].to_numpy(float)
    fired = (d["phase2_outcome"].astype(str).to_numpy() == "activation")
    act, dep, hyp = states(dv, fired, neu)
    cm_dep, cm_hyp = _trunc("Reds", 0.30, 1.0), _trunc("Blues_r", 0.0, 0.70)
    n_dep, n_hyp = Normalize(neu, dvm_max), Normalize(-dvm_max, -neu)
    spec = sorted(d["morphology"].astype(str).unique())
    grey = {m: GRAY_LEVELS[i % len(GRAY_LEVELS)] for i, m in enumerate(spec)}
    morph_morph = d["morphology"].astype(str).to_numpy()
    n = len(d)
    # dense cultures (1700 neurons): lighter morphologies and smaller somata, else the figure is a grey carpet
    soma_size = soma_size if soma_size is not None else (110.0 if n <= 300 else 40.0)
    morph_alpha = morph_alpha if morph_alpha is not None else (0.6 if n <= 300 else 0.2)
    panels = (("act", act, f"Activated  (whole simulated window)\n{int(act.sum())} of {n} neurons"),
              ("dep", dep, f"Depolarized at t0\n{int(dep.sum())} of {n} neurons"),
              ("hyp", hyp, f"Hyperpolarized at t0\n{int(hyp.sum())} of {n} neurons"))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    for ax, (key, m, ttl) in zip(axes, panels):
        if polys:
            for mname in polys:
                sel = np.where((morph_morph == mname) & np.isfinite(theta))[0]
                segs = [rotate_translate(p, xy[i, 0], xy[i, 1], theta[i]) for i in sel for p in polys[mname]]
                if segs:
                    ax.add_collection(LineCollection(segs, colors=grey[mname], linewidths=0.3, alpha=morph_alpha, zorder=2,
                                                     rasterized=True))   # keeps the PDF small (500k lines otherwise)
        for (ex, ey) in electrodes(pitch):
            ax.add_patch(Rectangle((ex - ELEC_UM / 2, ey - ELEC_UM / 2), ELEC_UM, ELEC_UM, fc="none",
                                   ec="0.25", lw=0.9, zorder=6))
        ax.scatter(xy[~m, 0], xy[~m, 1], s=9, c="0.82", linewidths=0, zorder=3)
        if key == "act":
            ax.scatter(xy[m, 0], xy[m, 1], s=soma_size, c=[GREEN], edgecolors="k", linewidths=0.8, zorder=5)
        else:
            cm, nm = (cm_dep, n_dep) if key == "dep" else (cm_hyp, n_hyp)
            ax.scatter(xy[m, 0], xy[m, 1], s=soma_size, c=dv[m], cmap=cm, norm=nm, edgecolors="k",
                       linewidths=0.8, zorder=5)
            fig.colorbar(ScalarMappable(nm, cm), ax=ax, fraction=0.046, pad=0.03, label="\u0394Vm at t0 (mV)")
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=11); ax.set_xlabel("x (\u00b5m)")
    axes[0].set_ylabel("y (\u00b5m)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    files = [out_stem + ".png", out_stem + ".pdf"]
    fig.savefig(files[0], dpi=300); fig.savefig(files[1])
    plt.close(fig)
    return files, (int(act.sum()), int(dep.sum()), int(hyp.sum()))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the built-in checks and exit")
    ap.add_argument("--input", "--merged", dest="merged",
                    help="ONE raw culture_Pactivation*.csv of culture_export (no merge needed) or a merged_*.csv")
    ap.add_argument("--list", action="store_true", help="list cultures and neurons per layer, then exit")
    ap.add_argument("--layer", type=float, default=80.0, help="slab thickness (um): 40, 80 or 120")
    ap.add_argument("--culture", default=None, help="culture name, e.g. S1000_C0 (see --list; default: first in the file)")
    ap.add_argument("--pool", action="store_true", help="pool all cultures (no morphologies)")
    ap.add_argument("--neu", type=float, default=1.0, help="threshold (mV) for depolarized / hyperpolarized")
    ap.add_argument("--dvm-max", type=float, default=10.0, help="colour-scale limit (mV)")
    ap.add_argument("--half", type=float, default=None, help="half-width of the maps (um); default: data extent")
    ap.add_argument("--soma-size", type=float, default=None, help="soma marker area; default 110 (<=300 neurons) or 40")
    ap.add_argument("--morph-alpha", type=float, default=None, help="opacity of the morphologies; default 0.6 (<=300 neurons) or 0.2")
    ap.add_argument("--out", default="figures/culture_states")
    ap.add_argument("--seed", type=int, default=None, help="campaign seed (default: the `seed` column of a raw file)")
    ap.add_argument("--span", type=float, default=None, help="half side S of the somata square (um); default: inferred")
    ap.add_argument("--center", type=float, nargs=2, default=None, metavar=("CX", "CY"),
                    help="centre of the somata square (um); default: inferred\n"
                         "((0, 0) before 2026-09-28, the dipole centre (0, -30) since)")
    ap.add_argument("--n-neurons", type=int, default=None, help="neurons per culture; default: inferred")
    ap.add_argument("--n-morph", type=int, default=None, help="number of specimens; default: inferred")
    ap.add_argument("--morph-dir", default=None, help="folder searched for specimen_*/morphology.asc")
    ap.add_argument("--asc", action="append", default=[], metavar="SPEC=PATH")
    ap.add_argument("--require-all-morph", action="store_true",
                    help="stop with an error if the .asc of ANY specimen of the culture is missing "
                         "(guarantees a complete figure)")
    ap.add_argument("--no-orient-check", action="store_true",
                    help="check the rebuilt rotations through the positions only, not theta_orient_deg\n"
                         "(for a file whose theta_orient_deg follows another definition; wrong rotations\n"
                         "for a culture cut short -- a job still running or killed)")
    ap.add_argument("--find-seed", action="store_true", help="search the seed that reproduces the table positions")
    ap.add_argument("--seed-range", type=int, nargs=2, default=[0, 100000])
    ap.add_argument("--chunksize", type=int, default=500_000)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.merged:
        ap.error("--input is required (or --selftest)")
    paths = sorted(glob.glob(a.merged)) or [a.merged]
    path = paths[0]
    if len(paths) > 1:
        print(f"NOTE: several files match, using {path} (one file is enough: pass ONE shard)")
    if a.list:
        acc = sorted(list_cultures(path, a.chunksize).items())
        names = sorted({k[0] for k, _ in acc})
        layers = sorted({k[1] for k, _ in acc})
        sizes = sorted({n for _, n in acc})
        print(f"{len(names)} cultures, layers {layers}, neurons per (culture, layer): {sizes[:5]}")
        for (cg, layer), n in acc[:30]:
            print(f"{cg}  layer {layer}: {n} neurons")
        if len(acc) > 30:
            print(f"... ({len(acc) - 30} more lines)")
        return None

    d = _read_filtered(path, a.layer, a.culture, not a.pool, a.chunksize)
    if d.empty:
        raise SystemExit(f"no rows for layer {a.layer:g} / culture {a.culture}; try --list")
    d = d.sort_values(["culture_global", "neuron"]).reset_index(drop=True)
    cult = d["culture_global"].iloc[0]

    if a.find_seed:
        if None in (a.n_neurons, a.span, a.n_morph):
            raise SystemExit("--find-seed needs --n-neurons, --span and --n-morph")
        hits = find_seed(d, a.n_neurons, a.n_morph, a.span, *a.seed_range)
        print("seeds reproducing the positions:", hits if hits else "none in the range (check span / N / n-morph)")
        return hits

    half = a.half or float(np.ceil(max(np.abs(d["x_um"]).max(), np.abs(d["y_um"]).max()) / 50.0) * 50.0)
    theta, polys, note = np.full(len(d), np.nan), {}, ""
    if a.seed is not None or a.morph_dir or a.asc:
        try:
            if a.pool:
                raise ValueError("morphologies are drawn for one culture, not with --pool")
            if a.seed is None and "seed" not in d.columns:
                raise ValueError("no `seed` column in this file: give --seed (or find it with --find-seed)")
            n_neu, n_mor, spn = a.n_neurons, a.n_morph, a.span
            cen = None if a.center is None else (float(a.center[0]), float(a.center[1]))
            given = (n_neu, n_mor, spn, cen)
            if None in given:
                i_n, i_m, i_s, i_c = infer_params(d, a.seed, n_neurons=n_neu, span=spn,
                                                  check_orient=not a.no_orient_check)
                n_neu, n_mor, spn, cen = n_neu or i_n, n_mor or i_m, spn or i_s, cen or i_c
            print(f"  parameters: neurons per culture N={n_neu}, specimens={n_mor}, half side S={spn:g} um, "
                  f"centre ({cen[0]:g}, {cen[1]:g}) um" + ("  (inferred)" if None in given else ""))
            asc_map = {}
            for item in a.asc:
                k, _, v = item.partition("=")
                asc_map[k.strip()] = os.path.expanduser(v.strip())
            spec = sorted(d["morphology"].astype(str).unique())
            if a.morph_dir:
                for m in spec:
                    if m not in asc_map and find_asc(m, a.morph_dir):
                        asc_map[m] = find_asc(m, a.morph_dir)
            asc_map = {m: p for m, p in asc_map.items() if os.path.isfile(p)}
            if not asc_map:
                raise ValueError("no morphology file found")
            theta = rotations_from_seed(d, a.seed, n_neu, n_mor, spn, cen, check_orient=not a.no_orient_check)
            if a.no_orient_check or "theta_orient_deg" not in d.columns:
                print("  NOTE: rotations checked through the positions only ("
                      + ("--no-orient-check" if a.no_orient_check else "no theta_orient_deg column")
                      + "): right for a complete culture, wrong for one cut short")
            polys = {m: sliced_polylines(p, a.layer) for m, p in asc_map.items() if m in spec}
            missing = [m for m in spec if m not in polys]
            if missing and a.require_all_morph:
                raise SystemExit(f"STOP: no .asc found for specimen(s) {missing} (found {sorted(polys)}). "
                                 f"Put specimen_*/morphology.asc of ALL specimens under --morph-dir, "
                                 f"or give --asc SPEC=PATH for each.")
            print(f"  morphologies drawn for {sorted(polys)}" + (f"; NOT available for {missing}" if missing else ""))
            if missing:
                note = f"   (morphologies drawn for {', '.join(sorted(polys))} only)"
        except Exception as e:                                   # noqa: BLE001
            print(f"  NOTE: morphologies not drawn ({type(e).__name__}: {e}). Somata only.")
            polys = {}

    title = (f"{d['culture_global'].nunique()} simulated cultures pooled \u2014 layer {a.layer:.0f} \u00b5m" if a.pool
             else f"Culture {cult} \u2014 layer {a.layer:.0f} \u00b5m") + note
    files, cnt = draw_figure(d, theta, polys, a.layer, half, a.neu, a.dvm_max, a.soma_size, title, a.out,
                             morph_alpha=a.morph_alpha)
    # the log line in plain ASCII (any locale can print it); the figure keeps the typographic title
    log_title = title.replace("\u2014", "--").replace("\u00b5", "u")
    print(f"{log_title}: {len(d)} neurons -> activated {cnt[0]}, depolarized {cnt[1]}, hyperpolarized {cnt[2]}, "
          f"neutral {len(d) - sum(cnt)}  (threshold {a.neu:g} mV)\nwritten: {', '.join(files)}")
    return files, cnt


# ----------------------------------------------------------------------------- self test
def selftest():
    """Built-in checks on synthetic data (no project files needed)."""
    results, tmp = [], tempfile.mkdtemp(prefix="cstates_")

    def check(name, fn):
        try:
            fn(); results.append((name, True, ""))
        except Exception as e:                                   # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {e}"))

    def synth(seed=7, span=150.0, n=30, n_morph=3, ncult=2):
        rows = []
        names = ["60308", "130303", "60303"]
        for c in range(ncult):
            midx, pos, th = draws(seed, c, n, n_morph, span)
            rng = np.random.default_rng(1000 + c)
            dv = np.concatenate([[5.0, -5.0, 0.5, -0.5, 1.0, -1.0], rng.normal(0, 4, n - 6)])
            sp = np.zeros(n, bool); sp[[0, 1, 2]] = True
            lab = np.where(sp, "activation", np.where(dv > 0, "depol", "hyperpol"))
            rows.append(pd.DataFrame({"culture_global": f"F00000_C{c}", "neuron": np.arange(n), "layer_um": 80,
                                      "morphology": [names[i] for i in midx], "x_um": pos[:, 0].round(2),
                                      "y_um": pos[:, 1].round(2),
                                      "deltaVm_end_phase2_mV": np.where(sp, 120.0, dv), "phase2_outcome": lab}))
        return pd.concat(rows, ignore_index=True)

    def synth_raw(seed=11, span=500.0, n=300, n_morph=6, ncult=2, center=(0.0, 0.0)):
        """Raw culture_export layout: `culture` + `seed`, no `culture_global`; all three layers."""
        names = ["60308", "130303", "60303", "60311", "130305", "130306"]
        rows = []
        for c in range(ncult):
            midx, pos, th = draws(seed, c, n, n_morph, span, center)
            rng = np.random.default_rng(2000 + c)
            sp = rng.random(n) < 0.05
            dv = np.where(sp, 120.0, rng.normal(0, 4, n))
            for layer in (40, 80, 120):
                rows.append(pd.DataFrame({"culture": c, "neuron": np.arange(n), "morphology": [names[i] for i in midx],
                                          "layer_um": layer, "x_um": pos[:, 0].round(2), "y_um": pos[:, 1].round(2),
                                          "theta_orient_deg": np.round(fold_orientation(th), 1),
                                          "fired": sp.astype(int), "seed": seed,
                                          "deltaVm_end_phase2_mV": dv,
                                          "phase2_outcome": np.where(sp, "activation", np.where(dv > 0, "depol", "hyperpol"))}))
        return pd.concat(rows, ignore_index=True)

    T = synth()
    csv = os.path.join(tmp, "merged.csv"); T.to_csv(csv, index=False)
    TR = synth_raw()
    rawcsv = os.path.join(tmp, "culture_Pactivation_raw.csv"); TR.to_csv(rawcsv, index=False)

    def t1():
        a, dd, h = states(np.array([5, -5, 0.5, -0.5, 1.0, -1.0, 0.999]), np.array([0, 0, 0, 0, 0, 0, 0]), 1.0)
        assert dd.tolist() == [True, False, False, False, True, False, False]
        assert h.tolist() == [False, True, False, False, False, True, False] and not a.any()
        a, dd, h = states(np.array([5.0, -5.0]), np.array([1, 1]), 1.0)
        assert a.all() and not dd.any() and not h.any()                      # spike wins

    def t2():
        got = _read_filtered(csv, 80, "F00000_C1", False, 7)
        assert len(got) == 30 and (got.culture_global == "F00000_C1").all()
        first = _read_filtered(csv, 80, None, True, 7)
        assert first.culture_global.nunique() == 1 and first.culture_global.iloc[0] == "F00000_C0"
        assert len(_read_filtered(csv, 80, None, False, 7)) == 60
        assert list_cultures(csv, 11) == {("F00000_C0", 80): 30, ("F00000_C1", 80): 30}

    def t3():
        d = _read_filtered(csv, 80, None, False, 100).sort_values(["culture_global", "neuron"]).reset_index(drop=True)
        th = rotations_from_seed(d, 7, 30, 3, 150.0)
        assert np.allclose(th[:30], draws(7, 0, 30, 3, 150.0)[2]) and np.allclose(th[30:], draws(7, 1, 30, 3, 150.0)[2])
        for bad in ((8, 30, 3, 150.0), (7, 30, 3, 160.0)):
            try:
                rotations_from_seed(d, *bad); raise AssertionError("accepted a wrong seed/span")
            except ValueError:
                pass
        assert 7 in find_seed(d, 30, 3, 150.0, 0, 50)

    def t4():
        files, cnt = main(["--merged", csv, "--layer", "80", "--culture", "F00000_C0", "--out", os.path.join(tmp, "o", "s")])
        assert os.path.getsize(files[0]) > 30_000
        sp = T[(T.culture_global == "F00000_C0")]
        v = sp.deltaVm_end_phase2_mV.to_numpy(); f = (sp.phase2_outcome == "activation").to_numpy()
        assert cnt == (int(f.sum()), int(((~f) & (v >= 1)).sum()), int(((~f) & (v <= -1)).sum()))

    def t5():
        try:
            import morphio  # noqa: F401  (availability check)
        except ImportError:
            print("   (morphio not installed: slicing test skipped; needed only for --morph-dir)"); return
        swc = os.path.join(tmp, "m.swc")
        with open(swc, "w") as f:
            f.write("1 1 0 0 0 5 -1\n")                                      # soma
            f.write("2 3 0 0 0 1 1\n3 3 20 0 2 1 2\n4 3 40 0 4 1 3\n")         # flat dendrite A (kept always)
            f.write("5 3 0 0 0 1 1\n6 3 30 0 27 1 5\n7 3 60 0 55 1 6\n")        # dendrite B: rises to z = 55
        sc, tree = _read_tree(swc)
        K = [kept_ids(tree, sc[2], L) for L in (40, 80, 120)]
        assert K[0] <= K[1] <= K[2], "nesting violated"
        for k in K:
            for i in k:
                assert tree[i]["parent"] is None or tree[i]["parent"] in k, "connectivity broken"
        # B has 20/55 of its arc inside +/-20 um and 40/55 inside +/-40 um (both < 0.8): dropped at 40 and 80,
        # fully inside +/-60 um: kept at 120
        assert len(K[0]) == len(K[1]) == 1 and len(K[2]) == 2, [len(k) for k in K]
        p = sliced_polylines(swc, 120)
        assert len(p) == 2 and all(q.shape[1] == 2 for q in p)

    def t7():
        try:
            import morphio  # noqa: F401  (availability check)
        except ImportError:
            print("   (morphio not installed: skipped)"); return
        swc = os.path.join(tmp, "m.swc")                       # written by t5
        args = ["--merged", csv, "--layer", "80", "--culture", "F00000_C0", "--seed", "7", "--span", "150",
                "--n-neurons", "30", "--n-morph", "3", "--asc", "60308=" + swc, "--out", os.path.join(tmp, "o7", "s")]
        main(args)                                              # incomplete figure is allowed (title says so)
        try:
            main(args + ["--require-all-morph"]); raise AssertionError("did not stop on missing specimens")
        except SystemExit as e:
            assert "STOP" in str(e.code)

    def t8():
        d = _read_filtered(rawcsv, 80, None, True, 50)
        assert d.culture_global.unique().tolist() == ["S11_C0"] and len(d) == 300
        assert sorted(list_cultures(rawcsv, 100)) == [("S11_C0", 40), ("S11_C0", 80), ("S11_C0", 120),
                                                      ("S11_C1", 40), ("S11_C1", 80), ("S11_C1", 120)]
        N, M, S, C = infer_params(d)
        assert (N, M, S, C) == (300, 6, 500.0, (0.0, 0.0)), (N, M, S, C)
        d2 = _read_filtered(rawcsv, 80, None, False, 50).sort_values(["culture_global", "neuron"]).reset_index(drop=True)
        th = rotations_from_seed(d2, None, 300, 6, 500.0)                 # seed taken from the table
        assert np.allclose(th[:300], draws(11, 0, 300, 6, 500.0)[2]) and np.allclose(th[300:], draws(11, 1, 300, 6, 500.0)[2])
        # S inference on other halves / seeds
        for sd, sp_ in ((5, 180.0), (123, 250.0)):
            T2 = synth_raw(seed=sd, span=sp_, n=120, n_morph=6, ncult=1)
            T2.to_csv(os.path.join(tmp, "r2.csv"), index=False)
            dd = _read_filtered(os.path.join(tmp, "r2.csv"), 80, None, True, 1000)
            assert infer_params(dd) == (120, 6, sp_, (0.0, 0.0))

    def t9():
        try:
            import morphio  # noqa: F401  (availability check)
        except ImportError:
            print("   (morphio not installed: skipped)"); return
        swc = os.path.join(tmp, "m.swc")
        out = main(["--input", rawcsv, "--layer", "80", "--culture", "S11_C1", "--asc", "60308=" + swc,
                    "--out", os.path.join(tmp, "o9", "s")])             # seed / N / S inferred from the raw table
        assert os.path.getsize(out[0][0]) > 30_000
        try:
            main(["--input", rawcsv, "--layer", "80", "--culture", "S11_C1", "--asc", "60308=" + swc,
                  "--require-all-morph", "--out", os.path.join(tmp, "o9", "s2")]); raise AssertionError("no stop")
        except SystemExit as e:
            assert "STOP" in str(e.code)

    def t10():
        """merged_*.csv of the real campaign: culture_global = <model>_S<seed>_C<c>, extra merge columns."""
        m = TR.copy()
        m["culture_global"] = "full_active_S11_C" + m["culture"].astype(str)
        m["source_file"] = "/x/culture_Pactivation_part0.csv"; m["source_index"] = 0
        m["neuron_global"] = m["culture_global"] + "_N" + m["neuron"].astype(str)
        mp = os.path.join(tmp, "merged_activation.csv"); m.to_csv(mp, index=False)
        d = _read_filtered(mp, 80, "full_active_S11_C1", False, 100)
        assert len(d) == 300 and d.culture_global.unique().tolist() == ["full_active_S11_C1"]
        assert infer_params(d) == (300, 6, 500.0, (0.0, 0.0))              # seed from the `seed` column
        th = rotations_from_seed(d.sort_values("neuron").reset_index(drop=True), None, 300, 6, 500.0)
        assert np.allclose(th, draws(11, 1, 300, 6, 500.0)[2])             # c parsed from the name suffix
        first = _read_filtered(mp, 80, None, True, 100)
        assert first.culture_global.iloc[0] == "full_active_S11_C0"
        assert len(list_cultures(mp, 100)) == 6

    def t11():
        """Square of the campaigns since 2026-09-28: +/-300 um around the dipole centre (0, -30)."""
        TC = synth_raw(seed=50000, span=300.0, n=200, n_morph=6, ncult=2, center=(0.0, -30.0))
        pc = os.path.join(tmp, "part_centred.csv"); TC.to_csv(pc, index=False)
        d = _read_filtered(pc, 80, "S50000_C1", False, 1000).sort_values("neuron").reset_index(drop=True)
        assert infer_params(d) == (200, 6, 300.0, (0.0, -30.0)), infer_params(d)
        assert infer_params(d, span=300.0)[2:] == (300.0, (0.0, -30.0))     # centre alone, S given
        th = rotations_from_seed(d, None, 200, 6, 300.0, (0.0, -30.0))
        assert np.allclose(th, draws(50000, 1, 200, 6, 300.0, (0.0, -30.0))[2])
        for bad in ((0.0, 0.0), (0.0, -29.0)):                               # a wrong centre is refused
            try:
                rotations_from_seed(d, None, 200, 6, 300.0, bad); raise AssertionError("accepted a wrong centre")
            except ValueError:
                pass
        assert 50000 in find_seed(d, 200, 6, 300.0, 49990, 50010)            # the centre cancels in the search
        T3 = synth_raw(seed=77, span=571.75, n=150, n_morph=6, ncult=1, center=(12.5, -37.5))
        p3 = os.path.join(tmp, "part_offgrid.csv"); T3.to_csv(p3, index=False)
        assert infer_params(_read_filtered(p3, 80, None, True, 1000)) == (150, 6, 571.75, (12.5, -37.5))
        try:
            import morphio  # noqa: F401  (availability check)
        except ImportError:
            print("   (morphio not installed: end-to-end part skipped)"); return
        swc = os.path.join(tmp, "m.swc")                                     # written by t5
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["--input", pc, "--layer", "80", "--culture", "S50000_C1", "--require-all-morph",
                  "--out", os.path.join(tmp, "o11", "s")]
                 + [x for m in ("60308", "130303", "60303", "60311", "130305", "130306") for x in ("--asc", m + "=" + swc)])
        log = out.getvalue()
        assert "centre (0, -30) um" in log and "morphologies drawn for" in log and "NOTE" not in log, log

    def t12():
        """culture_P*.csv written by culture_merge: `culture` renumbered 0..K-1 over the (seed, culture) pairs,
        the per-job index in `local_culture` -- the names and the draws must use local_culture."""
        A, B = synth_raw(seed=11, ncult=2), synth_raw(seed=1011, ncult=2)
        A["local_culture"], B["local_culture"] = A["culture"], B["culture"]
        B["culture"] = B["culture"] + 2
        mp = os.path.join(tmp, "culture_Pactivation_merge.csv"); pd.concat([A, B]).to_csv(mp, index=False)
        assert sorted({k[0] for k in list_cultures(mp, 500)}) == ["S1011_C0", "S1011_C1", "S11_C0", "S11_C1"]
        d = _read_filtered(mp, 80, "S1011_C1", False, 500).sort_values("neuron").reset_index(drop=True)
        assert len(d) == 300 and (d["culture"] == 3).all()                   # global id 3, per-job index 1
        assert infer_params(d) == (300, 6, 500.0, (0.0, 0.0))
        th = rotations_from_seed(d, None, 300, 6, 500.0)
        assert np.allclose(th, draws(1011, 1, 300, 6, 500.0)[2])
        # a merged_*.csv of culture_statistics over a culture_merge output AND a file without local_culture:
        # culture_global names every row, local_culture is blank for some -- nothing may be dropped
        M = pd.concat([A, B.drop(columns="local_culture")], ignore_index=True)
        M["culture_global"] = "full_tuned_S" + M["seed"].astype(str) + "_C" + M["culture"].astype(str)
        M.loc[:len(A) - 1, "culture_global"] = "full_tuned_S11_C" + A["local_culture"].astype(str)
        mm = os.path.join(tmp, "merged_activation_mixed.csv"); M.to_csv(mm, index=False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            n_rows = sum(list_cultures(mm, 500).values())
        assert n_rows == len(M) and "dropped" not in out.getvalue(), (n_rows, len(M), out.getvalue())

    def t13():
        """Cultures cut short (a job still running, or killed) and a half-written last row."""
        n_full, cut = 300, 290
        # a seed for which N=290 reproduces EVERY position of the 300-neuron culture: only the rotations
        # (drawn after the 2N position values) tell the two culture sizes apart
        sd = next(s for s in range(100, 1000)
                  if np.allclose(draws(s, 0, cut, 6, 300.0, (0.0, -30.0))[1],
                                 draws(s, 0, n_full, 6, 300.0, (0.0, -30.0))[1][:cut], atol=0.011))
        T = synth_raw(seed=sd, span=300.0, n=n_full, ncult=1, center=(0.0, -30.0))
        th_true = draws(sd, 0, n_full, 6, 300.0, (0.0, -30.0))[2]
        for keep_n in (cut, 120):
            p = os.path.join(tmp, f"part_cut{keep_n}.csv"); T[T.neuron < keep_n].to_csv(p, index=False)
            d = _read_filtered(p, 80, None, True, 1000).sort_values("neuron").reset_index(drop=True)
            assert infer_params(d) == (n_full, 6, 300.0, (0.0, -30.0)), (keep_n, infer_params(d))
            assert np.allclose(rotations_from_seed(d, None, n_full, 6, 300.0, (0.0, -30.0)), th_true[:keep_n])
        d = _read_filtered(os.path.join(tmp, f"part_cut{cut}.csv"), 80, None, True, 1000)
        d = d.sort_values("neuron").reset_index(drop=True)
        try:
            rotations_from_seed(d, None, cut, 6, 300.0, (0.0, -30.0))
            raise AssertionError("accepted N=290 for a 300-neuron culture")
        except ValueError as e:
            assert "theta_orient_deg" in str(e), e
        p = os.path.join(tmp, "part_running.csv"); T.to_csv(p, index=False)
        whole, cl_whole = _read_filtered(p, 80, None, True, 1000), list_cultures(p, 64)
        with open(p, "a") as fh:
            fh.write("0,300,60308,80,12.3")                    # a row cut while being written: no newline
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got, cl = _read_filtered(p, 80, None, True, 64), list_cultures(p, 64)
        assert len(got) == len(whole) and cl == cl_whole and "incomplete" in out.getvalue(), out.getvalue()
        q = os.path.join(tmp, "part_growing.csv"); T.to_csv(q, index=False)
        g = _csv_chunks(q, {"culture", "neuron", "layer_um", "seed"}, 100)
        n_read = len(next(g))                                  # snapshot taken, reader open ...
        T.to_csv(q, index=False, header=False, mode="a")       # ... while the job appends more rows
        assert n_read + sum(len(ch) for ch in g) == len(T)

    def t14():
        """theta_orient_deg of another definition: refused by default; --no-orient-check uses the positions."""
        global N_SCAN
        n_scan, N_SCAN = N_SCAN, 200                           # a short scan: same logic, faster test
        try:
            _t14_body()
        finally:
            N_SCAN = n_scan

    def _t14_body():
        T = synth_raw(seed=50000, span=300.0, n=120, ncult=1, center=(0.0, -30.0))
        T["theta_orient_deg"] = (T["theta_orient_deg"] + 30.0) % 90.0
        p = os.path.join(tmp, "part_other_orient.csv"); T.to_csv(p, index=False)
        d = _read_filtered(p, 80, None, True, 1000).sort_values("neuron").reset_index(drop=True)
        try:
            infer_params(d); raise AssertionError("accepted rotations that contradict theta_orient_deg")
        except OrientationMismatch as e:
            assert "--no-orient-check" in str(e), e
        assert infer_params(d, check_orient=False) == (120, 6, 300.0, (0.0, -30.0))
        th = rotations_from_seed(d, None, 120, 6, 300.0, (0.0, -30.0), check_orient=False)
        assert np.allclose(th, draws(50000, 0, 120, 6, 300.0, (0.0, -30.0))[2])
        try:
            import morphio  # noqa: F401  (availability check)
        except ImportError:
            print("   (morphio not installed: end-to-end part skipped)"); return
        swc = os.path.join(tmp, "m.swc")                                     # written by t5
        for extra, drawn in (([], False), (["--no-orient-check"], True)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["--input", p, "--layer", "80", "--asc", "60308=" + swc, "--out", os.path.join(tmp, "o14", "s")]
                     + extra)
            log = out.getvalue()
            assert ("morphologies drawn for" in log) == drawn, log
            assert ("rotations checked through the positions only" in log) == drawn, log
            assert ("never the rotations" in log) == (not drawn), log

    def t6():
        pts = np.array([[10.0, 0.0], [20.0, 0.0]])
        q = rotate_translate(pts, 5.0, 5.0, 90.0)
        assert np.allclose(q, [[5.0, 15.0], [5.0, 25.0]])

    for i, fn in enumerate((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14), 1):
        check(f"{i} {fn.__name__}: " + {1: "states: spike wins, >= / <= threshold", 2: "chunked reading and filters",
              3: "rotations rebuilt from the seed; wrong seed refused; find_seed", 4: "figure written, counts match by hand",
              5: "slicing: nested, connected, 2D polylines", 6: "rotation convention",
              7: "--require-all-morph stops when a specimen is missing",
              8: "raw culture_export CSV: names, list, inference of N / specimens / S, seed from the table",
              9: "raw CSV end to end with inferred parameters",
              10: "merged_*.csv of the campaign (<model>_S<seed>_C<c>): read, infer, rotations",
              11: "square centred on the dipole (0, -30): S and centre inferred, wrong centre refused, end to end",
              12: "culture_merge output (culture = global id): names and draws from local_culture",
              13: "culture cut short: culture size searched, checked on the rotations; half-written last row",
              14: "theta_orient_deg of another definition: refused; --no-orient-check uses the positions"}[i], fn)
    for n, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {n}  {msg}")
    nf = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - nf}/{len(results)} passed")
    shutil.rmtree(tmp, ignore_errors=True)
    if nf:
        sys.exit(1)


if __name__ == "__main__":
    main()
