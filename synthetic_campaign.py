"""synthetic_campaign.py -- fake worker parts with the real row schema, for OFFLINE tests.

No NEURON, no physics: every value is drawn so that it has the format, the sign conventions
and the blank pattern of a real full_tuned row (culture_export.CSV_HEADER), which is all the
merge / statistics / bump-statistics code looks at. Used by smoke_test_bump_statistics.py to
exercise the analysis at campaign scale (millions of rows) without simulating anything.

    from synthetic_campaign import write_parts
    write_parts("results_full_tuned/parts_fake", n_cultures=4, n_neurons=50, seed=10000)
"""
import csv
import os

import numpy as np

MORPHS = ("60308", "130303", "60303", "60311", "130305", "130306")
LAYERS = (40.0, 80.0, 120.0)


def _fmt(v, nd):
    """A number as the workers write it (rounded), or '' for a blank cell."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return repr(round(float(v), nd))


def culture_rows(seed, c, n_neurons, measured=True, rng=None):
    """Rows (lists in culture_export.CSV_HEADER order) of one culture: n_neurons neurons, each
    at the three layers, neuron-major like iter_culture_blocks. `measured` False writes the
    kinetics columns blank, as a culture outside config.bump_culture_fraction has them."""
    from culture_export import CSV_HEADER
    rng = np.random.default_rng([int(seed), int(c), 7]) if rng is None else rng
    rows = []
    for i in range(int(n_neurons)):
        morph = MORPHS[int(rng.integers(len(MORPHS)))]
        x, y = rng.uniform(-500.0, 500.0, 2)
        th_or = rng.uniform(-180.0, 180.0)
        d_cen = float(np.hypot(x, y))
        d_dip = float(np.sqrt(d_cen ** 2 + 100.0))
        th_pos = float(np.degrees(np.arctan2(y, x)))
        for layer in LAYERS:
            # response falls with distance; sign set by orientation (as in the real field)
            scale = 8.0 * np.exp(-d_dip / 90.0) * (1.0 + layer / 400.0)
            dv = scale * np.cos(np.radians(th_or)) + rng.normal(0.0, 1e-4)
            fired = int(rng.random() < 1.0 / (1.0 + np.exp((d_dip - 60.0) / 12.0)))
            if fired:
                label, dep, hyp = "activation", 0, 0
            elif abs(dv) <= 1e-6:
                label, dep, hyp = "neutral", 0, 0
            else:
                label, dep, hyp = ("depol", 1, 0) if dv > 0 else ("hyperpol", 0, 1)
            r = dict(culture=c, neuron=i, morphology=morph, layer_um=int(layer),
                     x_um=_fmt(x, 2), y_um=_fmt(y, 2), dist_nearest_elec_um=_fmt(d_cen * 0.9, 2),
                     dist_center_um=_fmt(d_cen, 2), dist_dipole3d_um=_fmt(d_dip, 2),
                     theta_orient_deg=_fmt(th_or, 1), theta_pos_deg=_fmt(th_pos, 1),
                     n_pulses=2, i0_uA="50.0", fired=fired, seed=int(seed),
                     cell_model="full_tuned", v_rest_mV="-74.2748", ctrl_drift_mV="-0.0",
                     deltaVm_end_phase2_mV=_fmt(dv, 6), phase2_outcome=label,
                     depolarized=dep, hyperpolarized=hyp)
            if measured:
                bump = 0.9 * scale / 8.0 * np.sign(dv)
                ok = int(abs(bump) >= 0.05 and not fired)
                tr, td = rng.normal(20.0, 2.0), rng.normal(95.0, 6.0)
                e_ok = int(abs(dv) > 1e-3)
                r.update(early_peak_mV=_fmt(dv * 1.05, 6), early_t_peak_ms=_fmt(0.025, 3),
                         early_sign=("depol" if dv > 0 else "hyperpol") if e_ok else "none",
                         early_tau_ms=_fmt(rng.uniform(0.1, 0.3), 4) if e_ok else "",
                         early_t_1e_ms=_fmt(rng.uniform(0.1, 0.2), 4) if e_ok else "",
                         early_amp_mV=_fmt(dv * 0.8, 6) if e_ok else "",
                         early_amp_over_peak=_fmt(0.8, 4) if e_ok else "",
                         early_t_fit_hi_ms=_fmt(0.2, 3) if e_ok else "",
                         early_r2=_fmt(rng.uniform(0.9, 1.0), 4) if e_ok else "",
                         early_fit_ok=e_ok, dexp_t0_ms="0.0", dexp_dv_t0_mV=_fmt(dv, 6),
                         dexp_peak_mV=_fmt(bump, 6), dexp_t_peak_ms=_fmt(rng.normal(95, 5), 2),
                         dexp_data_peak_mV=_fmt(bump * 0.98, 6),
                         dexp_data_t_peak_ms=_fmt(rng.normal(96, 5), 2),
                         dexp_tau_rise_ms=_fmt(tr, 3), dexp_tau_decay_ms=_fmt(td, 3),
                         dexp_tau_offset_ms=_fmt(0.165, 3), dexp_amp_mV=_fmt(bump * 1.6, 6),
                         dexp_offset_mV=_fmt(-dv * 0.1, 6),
                         dexp_sign=("depol" if bump > 0 else "hyperpol") if ok else "none",
                         dexp_r2=_fmt(rng.uniform(0.95, 1.0), 4), dexp_fit_ok=ok)
            else:
                r.update(early_sign="none", early_fit_ok=0, dexp_t0_ms="0.0",
                         dexp_sign="none", dexp_fit_ok=0)
            rows.append([r.get(k, "") for k in CSV_HEADER])
    return rows


def write_parts(parts_dir, n_cultures, n_neurons, seed, n_workers=2, measured_fraction=1.0):
    """Write parts_dir/part_NNN.csv like run_parallel.sh does (round-robin cultures over
    workers). Cultures are measured when culture_has_kinetics(seed, c, fraction) -- the same
    draw the worker uses. Returns the list of files."""
    from culture_export import CSV_HEADER, culture_has_kinetics
    os.makedirs(parts_dir, exist_ok=True)
    files = []
    for w in range(int(n_workers)):
        path = os.path.join(parts_dir, "part_%03d.csv" % w)
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(CSV_HEADER)
            for c in range(w, int(n_cultures), int(n_workers)):
                wk = culture_has_kinetics(seed, c, measured_fraction)
                wr.writerows(culture_rows(seed, c, n_neurons, measured=wk))
        files.append(path)
    return files
