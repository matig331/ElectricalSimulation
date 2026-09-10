"""
Smoke test for the PURE-PYTHON helpers in rich_validation.py.
Runs WITHOUT NEURON (imports only the offline helpers). This validates the analysis
math (spike->frequency, sustained/block detection, domain-split correlation, hotspot
ratio) so you can trust the numbers before spending a NEURON run.

Run:  python smoke_test_rich_validation.py
Expected: all lines print PASS and it exits 0.
"""
import numpy as np
from rich_validation import (first_isi_freq_hz, sustained, split_corr, hotspot_ratio,
                             pulse_onsets, aligned_windows, max_pulse_delta)

def _check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# --- first_isi_freq_hz: 20 ms ISI -> 50 Hz; uses ONLY first two spikes ---
_check("first_isi 20ms->50Hz", abs(first_isi_freq_hz([100, 120, 140, 999]) - 50.0) < 1e-9)
_check("first_isi <2 spikes->0", first_isi_freq_hz([100]) == 0.0 and first_isi_freq_hz([]) == 0.0)
_check("first_isi ignores later ISIs",
       abs(first_isi_freq_hz([0, 10, 500]) - 100.0) < 1e-9)   # 10 ms ISI -> 100 Hz

# --- sustained: spike in last 100 ms of the step -> not blocked ---
delay, dur = 200.0, 700.0                     # step = [200, 900] ms; tail = [800, 900]
_check("sustained: spike at 850ms -> True", sustained([210, 850], delay, dur) is True)
_check("sustained: only early spikes -> False (block)", sustained([210, 260, 310], delay, dur) is False)
_check("sustained: no spikes -> False", sustained([], delay, dur) is False)

# --- split_corr: dendrites correlated, axon = vertical stripe at x~0 ---
rng = np.random.default_rng(0)
xd = rng.uniform(-1, 1, 200); yd = 3 * xd + rng.normal(0, 0.2, 200)   # strong corr dendrites
xa = rng.normal(0, 1e-6, 200); ya = rng.uniform(-20, 20, 200)         # axon: x~0, y spread
x = np.concatenate([xd, xa]); y = np.concatenate([yd, ya])
is_axon = np.concatenate([np.zeros(200, bool), np.ones(200, bool)])
cc = split_corr(x, y, is_axon)
_check("split_corr: no_axon >> all (axon dilutes)", cc["no_axon"] > cc["all"] + 0.2)
_check("split_corr: no_axon strong (>0.9)", cc["no_axon"] > 0.9)
_check("split_corr: axon_only ~ 0 (vertical stripe)", abs(cc["axon_only"]) < 0.3)

# --- hotspot_ratio: gbar x100 inside [360,600] um -> ratio ~100 ---
d = np.linspace(0, 1000, 500)
g = np.where((d >= 360) & (d <= 600), 100.0, 1.0)
_check("hotspot_ratio ~100", abs(hotspot_ratio(d, g) - 100.0) < 1e-9)
g_flat = np.ones_like(d)
_check("hotspot_ratio flat ~1", abs(hotspot_ratio(d, g_flat) - 1.0) < 1e-9)

# --- pulse_onsets: t_on + k*ISI ---
on = pulse_onsets(100.0, 3, 5000.0)
_check("pulse_onsets 0.2Hz -> [100,5100,10100]", np.allclose(on, [100.0, 5100.0, 10100.0]))

# --- aligned_windows + max_pulse_delta: identical pulses -> delta ~0; drift -> delta>0 ---
dt = 0.025
t = np.arange(0.0, 15200.0, dt)                       # 3 pulses at 5000 ms ISI
v = np.full_like(t, -73.9)
onsets = pulse_onsets(100.0, 3, 5000.0)
for oi in onsets:                                     # identical Gaussian bump at each onset
    v += 40.0 * np.exp(-((t - (oi + 1.0)) ** 2) / (2 * 0.5 ** 2))
w = aligned_windows(t, v, onsets, pre_ms=5.0, win_ms=25.0)
_check("aligned_windows: 3 equal-length windows", len(w) == 3 and len({len(x) for x in w}) == 1)
_check("identical pulses -> max_pulse_delta ~0", max_pulse_delta(w) < 1e-6)

v2 = v.copy()                                         # make pulse 3 taller (drift)
v2 += 5.0 * np.exp(-((t - (onsets[2] + 1.0)) ** 2) / (2 * 0.5 ** 2))
w2 = aligned_windows(t, v2, onsets, pre_ms=5.0, win_ms=25.0)
_check("drifting pulse -> max_pulse_delta ~5 mV", abs(max_pulse_delta(w2) - 5.0) < 0.2)
_check("max_pulse_delta <2 windows -> 0", max_pulse_delta(w[:1]) == 0.0)

print("\nAll smoke tests passed.")
