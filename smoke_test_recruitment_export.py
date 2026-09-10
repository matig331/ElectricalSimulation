"""
Smoke test for the PURE helpers in recruitment_export.py (no NEURON).
Run:  python smoke_test_recruitment_export.py   -> expect all PASS, exit 0.
"""
import numpy as np
from recruitment_export import nearest_electrode_dist, p_over_theta, bin_map

def _check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# nearest_electrode_dist
elec = [(-30, -30), (-30, 30), (30, -30), (30, 30)]
_check("nearest at origin ~42.4", abs(nearest_electrode_dist(0, 0, elec) - np.hypot(30, 30)) < 1e-9)
_check("nearest on an electrode = 0", nearest_electrode_dist(30, 30, elec) == 0.0)

# p_over_theta
_check("p_over_theta [1,0,1,0] -> 0.5", p_over_theta([1, 0, 1, 0]) == 0.5)
_check("p_over_theta all fired -> 1", p_over_theta([1, 1, 1]) == 1.0)
_check("p_over_theta empty -> nan", np.isnan(p_over_theta([])))

# bin_map: points near center with P=1 land in the middle bin; empty bins are nan
xs = [0.0, 1.0, -1.0];  ys = [0.0, -1.0, 1.0];  ps = [1.0, 1.0, 1.0]
Z, ext = bin_map(xs, ys, ps, half=100.0, n_bins=5)
_check("bin_map shape 5x5", Z.shape == (5, 5))
_check("bin_map extent", ext == [-100.0, 100.0, -100.0, 100.0])
_check("bin_map center bin = 1.0", Z[2, 2] == 1.0)
_check("bin_map corner empty -> nan", np.isnan(Z[0, 0]))
# a bin with mixed 1/0 averages
Z2, _ = bin_map([10.0, 10.0], [10.0, 10.0], [1.0, 0.0], half=100.0, n_bins=5)
mid = np.digitize([10.0], np.linspace(-100, 100, 6))[0] - 1
_check("bin_map mixed bin = 0.5", Z2[mid, mid] == 0.5)

print("\nAll smoke tests passed.")
