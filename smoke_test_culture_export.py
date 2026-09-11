"""Offline smoke test for culture_export pure helpers (no NEURON).

    python smoke_test_culture_export.py      # expect the last line: 'All smoke tests passed.'

Sections 1-N (original): geometry, binning, logistic, per-culture helpers, legacy classifiers.
Section 'SOMA-ONLY MERGE' (new): schema, phase-2 outcome rule, sham subtraction, row builder,
per-outcome writer, vectorised per_soma_P == old implementation, culture_draws == the old inline
RNG sequence (bit for bit), cell-model validation, renderer display cap.
"""
import numpy as np
from culture_export import (electrode_center, dipole_axis_deg, dist_from_center,
                            dist_from_nearest_electrode, dipole_frame, directional_rt, smooth_P_field,
                            rel_orientation_deg, radial_bin_P, per_culture_P, logistic_decreasing,
                            fit_logistic_decreasing, assign_morphologies,
                            grid_P_2d, iso_radii, activation_area_per_culture, per_soma_P, classify_response, fit_peak_kinetics)

def _c(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}"); assert cond, name

elec = np.array([[-30,-30],[-30,30],[30,-30],[30,30]], float)
sign = np.array([1,1,-1,-1], float)   # anodes x=-30, cathodes x=+30
_c("center at origin", np.allclose(electrode_center(elec), [0,0]))
_c("dipole axis ~0deg (anode->cathode along +x)", abs(dipole_axis_deg(elec, sign)-0.0) < 1e-9)
_c("monopolar axis -> 0", dipole_axis_deg(elec, np.array([1,1,1,1.0]))==0.0)

_c("dist from center", np.allclose(dist_from_center([[3,4],[0,0]],[0,0]), [5,0]))

# nearest-electrode distance: point next to one electrode is ~0 even if far from centroid
E = np.array([[-60,30],[60,30],[0,-30.0]])
dn = dist_from_nearest_electrode([[60,30],[0,0]], E)
_c("nearest: on an electrode -> 0", abs(dn[0]) < 1e-9)
_c("nearest: origin -> min electrode dist", abs(dn[1]-min(np.hypot(E[:,0],E[:,1]))) < 1e-9)

# orientation relative to a 0deg dipole axis, folded to [0,90]
_c("rel orient 90", rel_orientation_deg([90],0.0)[0]==90.0)
_c("rel orient 170 -> 10", abs(rel_orientation_deg([170],0.0)[0]-10.0)<1e-9)
_c("rel orient 180 -> 0", abs(rel_orientation_deg([180],0.0)[0]-0.0)<1e-9)

# radial binning: near all fire, far none
d = np.array([5,6,7, 55,56,57]); f = np.array([1,1,1, 0,0,0])
ctr, P, cnt = radial_bin_P(d, f, np.array([0.,10.,50.,100.]))
_c("bin0 P=1", P[0]==1.0)
_c("bin2 P=0", P[2]==0.0)
_c("bin1 empty -> nan", np.isnan(P[1]) and cnt[1]==0)

# per-culture P: culture 0 all-fire near, culture 1 none near -> mean 0.5, sd 0.5 in that bin
d2 = np.array([5,5, 5,5]); f2 = np.array([1,1, 0,0]); cu = np.array([0,0, 1,1])
ctr, mean, sd, ncb, stack = per_culture_P(d2, f2, cu, np.array([0.,10.,50.]))
_c("per-culture mean bin0 = 0.5", abs(mean[0]-0.5) < 1e-9)
_c("per-culture sd bin0 = 0.5", abs(sd[0]-0.5) < 1e-9)
_c("per-culture n cultures/bin0 = 2", ncb[0]==2)

# activation area per culture: culture 0 half fire, culture 1 all fire; span=50 -> sampled=100^2=10000
cu2 = np.array([0,0,0,0, 1,1,1,1]); fr2 = np.array([1,1,0,0, 1,1,1,1])
cults, area, r_eq, amean, asd = activation_area_per_culture(cu2, fr2, span_um=50.0)
_c("area culture0 = 0.5*10000 = 5000", abs(area[0]-5000.0) < 1e-6)
_c("area culture1 = 1.0*10000 = 10000", abs(area[1]-10000.0) < 1e-6)
_c("mean area = 7500", abs(amean-7500.0) < 1e-6)
_c("r_eq matches sqrt(area/pi)", abs(r_eq[1]-np.sqrt(10000/np.pi)) < 1e-6)

# logistic decreasing: ~1 near 0, ~0 far, 0.5 at r50
_c("logistic ~1 near", logistic_decreasing(0,50,10) > 0.99)
_c("logistic 0.5 at r50", abs(logistic_decreasing(50,50,10)-0.5)<1e-9)
_c("logistic ~0 far", logistic_decreasing(120,50,10) < 0.01)

# all morphologies present
idx = assign_morphologies(30, 3, np.random.default_rng(0))
_c("all 3 morphologies present", set(idx.tolist())=={0,1,2})
_c("length preserved", len(idx)==30)

# numpy-only logistic fit recovers a known r50 (no scipy)
r = np.linspace(0, 120, 40)
p = logistic_decreasing(r, 55.0, 12.0)                 # synthetic ground truth
r50, w = fit_logistic_decreasing(r, p)
_c(f"fit recovers r50~55 (got {r50:.1f})", abs(r50-55.0) < 6.0)
_c(f"fit recovers w~12 (got {w:.1f})", abs(w-12.0) < 6.0)
# robustness: a SHARP drop must give a positive, in-range r50 (was the -55 bug)
rr = np.array([12,20,28,36,44,52,60,80,120.]); pp = np.array([1,1,1,0.5,0,0,0,0,0.])
r50s, ws = fit_logistic_decreasing(rr, pp)
_c(f"sharp drop -> r50 in range (got {r50s:.1f})", 0 < r50s < 60 and ws > 0)

# dipole frame: anodes x=-30, cathodes x=+30 -> centre x~0, direction points from - to + (+x -> -x)
elc = np.array([[-30,30],[-30,-30],[-30,-90],[30,30],[30,-30],[30,-90.]])
sgn = np.array([1,1,1,-1,-1,-1.])
dc, dd = dipole_frame(elc, sgn)
_c("dipole centre ~ x=0", abs(dc[0]) < 1e-9)
_c("dipole direction unit -x (cathode->anode)", abs(dd[0]+1.0) < 1e-9 and abs(dd[1]) < 1e-9)

# directional r/theta: toward anode (along +d) -> theta~0; perpendicular -> 90; toward cathode -> 180
r,th = directional_rt([[dc[0] + dd[0]*100, dc[1] + dd[1]*100]], dc, dd, z_um=0.0)
_c("toward-anode soma -> theta ~ 0", th[0] < 1e-6)
r2,th2 = directional_rt([[dc[0], dc[1]+100]], dc, dd, z_um=0.0)
_c("perpendicular soma -> theta ~ 90", abs(th2[0]-90.0) < 1e-6)
r3,th3 = directional_rt([[dc[0] - dd[0]*100, dc[1] - dd[1]*100]], dc, dd, z_um=0.0)
_c("toward-cathode soma -> theta ~ 180", abs(th3[0]-180.0) < 1e-6)
_c("3D distance includes z", directional_rt([[dc[0],dc[1]]], dc, dd, z_um=10.0)[0][0] == 10.0)

# phase_mask: p1 covers [t_on, t_on+phi), p2 covers [t_on+phi, t_on+2phi)
import field as _F
tt = np.arange(0, 6, 0.05)
m1 = _F.phase_mask(tt, 5.0, 0.25, "p1"); m2 = _F.phase_mask(tt, 5.0, 0.25, "p2"); mb = _F.phase_mask(tt, 5.0, 0.25, "both")
_c("phase_mask p1 window", abs(tt[m1].min()-5.0) < 0.06 and tt[m1].max() < 5.25)
_c("phase_mask p2 window", tt[m2].min() >= 5.25 and tt[m2].max() < 5.5)
_c("phase_mask both = all", mb.all())
_c("phase_mask p1/p2 disjoint", not (m1 & m2).any())

# smooth field: a cluster of firing somata near origin -> P high at origin, ~0 far
Xs=np.array([0,2,-2,1.]); Ys=np.array([0,1,-1,2.]); Fs=np.array([1,1,1,1.])
gx=np.array([0., 200.]); gy=np.array([0.])
Zf=smooth_P_field(Xs,Ys,Fs,gx,gy,bw=20.0)
_c("smooth field ~1 at cluster", Zf[0,0] > 0.99)
_c("smooth field ~0 far (nan)", np.isnan(Zf[0,1]) or Zf[0,1] < 0.01)

# 2D map: near-centre bin fires, far bin silent
xs = np.array([2,3,-2, 80,81]); ys = np.array([0,1,-1, 0,1]); fr = np.array([1,1,1, 0,0])
Z, ext = grid_P_2d(xs, ys, fr, half=100.0, n_bins=5)
_c("grid shape 5x5", Z.shape == (5,5))
_c("grid centre bin = 1.0", Z[2,2] == 1.0)
_c("grid empty corner -> nan", np.isnan(Z[0,0]))

# iso radii: P=0.5 crosses exactly at r50; higher P -> smaller radius
radii = dict((round(L,2), r) for L, r in iso_radii(50.0, 10.0, levels=(0.75,0.5,0.25)))
_c("iso r at P=0.5 == r50", abs(radii[0.5]-50.0) < 1e-9)
_c("iso r at P=0.75 < r50 < P=0.25", radii[0.75] < 50.0 < radii[0.25])

# classify_response: spike -> activation; dominant deflection sign -> depol/hyperpol; tiny -> none
_c("classify spike -> activation", classify_response(1, 20.0, 1.0)=="activation")
_c("classify depol (dep>hyp)", classify_response(0, 5.0, 1.0)=="depol")
_c("classify hyperpol (hyp>dep)", classify_response(0, 1.0, 8.0)=="hyperpol")
_c("classify none (tiny)", classify_response(0, 0.1, 0.2)=="none")

# fit_peak_kinetics: skip 7 ms, SIGNED peak (raw) in [7,200]; sign not forced positive
tt=np.arange(0,1000,0.025)
transient=20*np.exp(-tt/0.2)                               # big + capacitive transient (skipped)
bump_neg=-3.0*(tt/40.0)*np.exp(1-tt/40.0)                  # slow NEGATIVE bump (hyper), peaks ~40ms
y=transient+bump_neg
pk,tpk,td=fit_peak_kinetics(tt,y)
_c(f"signed peak HYPER despite +transient (pk={pk:.2f})", pk<0)
_c(f"peak time in [7,200] (got {tpk:.0f})", 7<=tpk<=200)
_c(f"tau_decay finite (got {td:.0f})", np.isfinite(td) and td>0)
# positive bump -> depol
pk2,_,td2=fit_peak_kinetics(tt, transient+3.0*(tt/40.0)*np.exp(1-tt/40.0))
_c("positive bump -> depol", pk2>0)
# a trace that goes MORE negative than positive in the window -> hyper (raw, not abs)
y3=transient + 1.0*np.exp(-tt/50.0) - 4.0*(tt/60.0)*np.exp(1-tt/60.0)
pk3,_,_=fit_peak_kinetics(tt,y3)
_c("more-negative bump -> hyper (raw sign)", pk3<0)


# =============================== SOMA-ONLY MERGE (new) =============================== #
import csv, os, tempfile
import culture_export as CE

print("\n--- schema")
_c("CSV_HEADER extends the legacy 15 columns in their ORIGINAL order",
   CE.CSV_HEADER[:15] == CE.LEGACY_CSV_HEADER and len(CE.LEGACY_CSV_HEADER) == 15)
_c("CSV_HEADER column names are unique", len(set(CE.CSV_HEADER)) == len(CE.CSV_HEADER))
_c("every outcome column is in CSV_HEADER", all(c in CE.CSV_HEADER for c in CE.OUTCOME_COLS))
_c("OUTCOME_FILES maps the three outcomes to the three columns",
   sorted(v[1] for v in CE.OUTCOME_FILES.values()) == sorted(CE.OUTCOME_COLS))

print("\n--- phase-2 outcome rule (sign of DeltaV at the END of phase 2 only)")
P2 = CE.phase2_end_outcome
_c("spike wins over any DeltaV", P2(1, -5.0) == ("activation", 1, 0, 0) and P2(1, 5.0)[0] == "activation")
_c("DeltaV > eps -> depol", P2(0, 2e-6) == ("depol", 0, 1, 0))
_c("DeltaV < -eps -> hyperpol", P2(0, -2e-6) == ("hyperpol", 0, 0, 1))
_c("|DeltaV| <= eps -> neutral", P2(0, 5e-7) == ("neutral", 0, 0, 0) and P2(0, 0.0)[0] == "neutral")
_c("nan DeltaV -> neutral (never a silent depol/hyperpol)", P2(0, float("nan"))[0] == "neutral")
_c("outcomes are mutually exclusive for any input",
   all(sum(P2(f, dv)[1:]) <= 1 for f in (0, 1) for dv in (-1, -1e-7, 0, 1e-7, 1, float("nan"))))

print("\n--- end-of-phase-2 sample")
tw = np.array([-0.05, -0.025, 0.0, 0.025]); vw = np.array([1.0, 2.0, 3.0, 4.0])
_c("picks the sample at t = 0", CE.v_at_end_of_phase2(tw, vw) == 3.0)
_c("nearest sample if 0 is not on the grid", CE.v_at_end_of_phase2(tw + 0.004, vw) == 3.0)
_c("empty window -> nan", np.isnan(CE.v_at_end_of_phase2([], [])))
_c("length mismatch -> nan", np.isnan(CE.v_at_end_of_phase2(tw, vw[:3])))

print("\n--- sham subtraction removes the init drift (the reason for the change)")
t = np.arange(0.0, 11.5 + 0.025, 0.025); t_end = 5.5; tw_ = t - t_end
v_rest = -83.18
drift = 0.0896 * (1 - np.exp(-t / 3.0)) / (1 - np.exp(-t_end / 3.0))   # +0.0896 mV at t_end
resp = -0.004 * np.exp(-((t - t_end) / 0.6) ** 2)                      # weak HYPERPOL response
v_sham = v_rest + drift
v_stim = v_rest + drift + resp
dv_sham = CE.v_at_end_of_phase2(tw_, v_stim) - CE.v_at_end_of_phase2(tw_, v_sham)
dv_scalar = CE.v_at_end_of_phase2(tw_, v_stim) - v_rest
_c(f"sham-referenced DeltaV recovers the response ({dv_sham:+.4f} mV)", abs(dv_sham - resp[220]) < 1e-12)
_c("sham-referenced sign = hyperpol (correct)", P2(0, dv_sham)[0] == "hyperpol")
_c(f"scalar-rest reference gives the WRONG sign here ({dv_scalar:+.4f} mV)", P2(0, dv_scalar)[0] == "depol")
_c("compat wrapper _phase2_end_sign agrees", CE._phase2_end_sign(tw_, v_stim, v_sham[220])[1] == "hyperpol")

print("\n--- row builder")
prep = dict(cell_model="soma_only", v_rest=-83.180535, ctrl_drift=0.0896)
res = dict(fired=0, dv_end=-1.4e-6, outcome="hyperpol", depolarized=0, hyperpolarized=1)
row = CE.build_row(3, 7, "60308", 40.0, 1.234, -5.678, 20.0, 30.0, 31.0, 12.34, 99.99, 36, 50.0, 1000, prep, res)
d = dict(zip(CE.CSV_HEADER, row))
_c("row length == header length", len(row) == len(CE.CSV_HEADER))
_c("identity fields in place", (d["culture"], d["neuron"], d["morphology"], d["layer_um"], d["seed"]) == (3, 7, "60308", 40, 1000))
_c("provenance fields in place", d["cell_model"] == "soma_only" and d["v_rest_mV"] == -83.1805)
_c("stored DeltaV keeps the label's sign at the 1e-6 threshold", d["deltaVm_end_phase2_mV"] < 0 and d["phase2_outcome"] == "hyperpol")
_c("outcome triplet in place", (d["fired"], d["depolarized"], d["hyperpolarized"]) == (0, 0, 1))
try:
    CE.build_row(0, 0, "m", 40, 0, 0, 0, 0, 0, 0, 0, 1, 50, 0, prep, {})
    _c("build_row rejects an incomplete result", False)
except KeyError:
    _c("build_row rejects an incomplete result", True)

print("\n--- per-outcome files")
op = CE.outcome_paths("/x/y/culture_Pactivation.csv")
_c("derived names", op == {"activation": "/x/y/culture_Pactivation.csv",
                           "depolarization": "/x/y/culture_Pdepolarization.csv",
                           "hyperpolarization": "/x/y/culture_Phyperpolarization.csv"})
_c("dry-run prefix kept", CE.outcome_paths("dryrun_Pactivation.csv")["depolarization"] == "dryrun_Pdepolarization.csv")
try:
    CE.outcome_paths("results.csv"); _c("a name without 'Pactivation' is rejected", False)
except ValueError:
    _c("a name without 'Pactivation' is rejected", True)
for col in CE.OUTCOME_COLS:
    oc = CE.outcome_columns(CE.CSV_HEADER, col)
    _c(f"'{col}' file has exactly ONE outcome column, last", oc[-1] == col and sum(c in CE.OUTCOME_COLS for c in oc) == 1)

tmp = tempfile.mkdtemp()
rows = [CE.build_row(0, i, "60308", 40, i, -i, 10, 20, 21, 5, 90, 36, 50, 1000, prep,
                     dict(zip(("fired", "dv_end", "outcome", "depolarized", "hyperpolarized"), r)))
        for i, r in enumerate([(1, 9.0, "activation", 0, 0), (0, 0.2, "depol", 1, 0),
                               (0, -0.3, "hyperpol", 0, 1), (0, 0.0, "neutral", 0, 0)])]
with CE.OutcomeWriter(CE.CSV_HEADER, os.path.join(tmp, "sub", "culture_Pactivation.csv")) as ow:
    for r in rows:
        ow.write(r)
got = {}
for name, path in ow.paths.items():
    with open(path, newline="") as fh:
        got[name] = list(csv.DictReader(fh))
_c("three files written (sub-directory created)", sorted(got) == sorted(CE.OUTCOME_FILES))
_c("activation column = fired", [r["fired"] for r in got["activation"]] == ["1", "0", "0", "0"])
_c("depolarization column = depolarized", [r["depolarized"] for r in got["depolarization"]] == ["0", "1", "0", "0"])
_c("hyperpolarization column = hyperpolarized", [r["hyperpolarized"] for r in got["hyperpolarization"]] == ["0", "0", "1", "0"])
_c("no file carries another outcome's column",
   "fired" not in got["depolarization"][0] and "depolarized" not in got["activation"][0])
_c("the same simulations in the same order in all three files",
   len({tuple((r["culture"], r["neuron"], r["layer_um"]) for r in v) for v in got.values()}) == 1)
leg = [[0, 0, "60308", 40, 1, 1, 1, 1, 1, 1, 1, 16, 50, 1, 1000]]
with CE.OutcomeWriter(CE.LEGACY_CSV_HEADER, os.path.join(tmp, "leg", "culture_Pactivation.csv")) as ow2:
    ow2.write(leg[0])
_c("legacy header -> activation file only", list(ow2.paths) == ["activation"])

print("\n--- vectorised per_soma_P == the previous implementation")
rng = np.random.default_rng(3)
Xs = np.round(rng.uniform(-50, 50, 300), 2); Ys = np.round(rng.uniform(-50, 50, 300), 2)
Xs = np.concatenate([Xs, Xs[:120], Xs[:60]]); Ys = np.concatenate([Ys, Ys[:120], Ys[:60]])
Fs = (rng.uniform(size=Xs.size) < 0.4).astype(float)
def _per_soma_P_old(X, Y, fired):
    key = np.round(np.stack([X, Y], 1), 2)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    return uniq[:, 0], uniq[:, 1], np.array([fired[inv == k].mean() for k in range(len(uniq))])
a_new, a_old = per_soma_P(Xs, Ys, Fs), _per_soma_P_old(Xs, Ys, Fs)
_c("same somata, same P (to 1e-12)", all(np.allclose(x, y, atol=1e-12, rtol=0) for x, y in zip(a_new, a_old)))

print("\n--- culture_draws == the inline sequence of every earlier version (bit for bit)")
E6 = np.array([[-30, 30], [-30, -30], [-30, -90], [30, 30], [30, -30], [30, -90]], float)
S6 = np.array([1, 1, 1, -1, -1, -1], float)
cen = electrode_center(E6); ax = dipole_axis_deg(E6, S6); dc, dd = dipole_frame(E6, S6)
ok = True
for seed_, c_ in ((0, 0), (0, 7), (1000, 3), (4000, 95)):
    dr = CE.culture_draws(seed_, c_, 25, 6, 500.0, E6, cen, dc, dd, ax, 10.0)
    r0 = np.random.default_rng(seed_ + c_)                      # the old inline code
    midx0 = assign_morphologies(25, 6, r0); pos0 = r0.uniform(-500.0, 500.0, size=(25, 2))
    th0 = r0.uniform(0, 360, size=25)
    ok &= np.array_equal(dr["midx"], midx0) and np.array_equal(dr["pos"], pos0) and np.array_equal(dr["theta"], th0)
    ok &= np.array_equal(dr["d_near"], dist_from_nearest_electrode(pos0, E6))
_c("same morphologies, positions, orientations and distances", bool(ok))

print("\n--- cell model")
_c("valid names accepted", CE.resolve_cell_model("soma_only") == "soma_only" and CE.resolve_cell_model("full_active") == "full_active")
for bad in ("somaonly", None):
    try:
        CE.resolve_cell_model(bad if bad else type("C", (), {})()); _c(f"invalid/missing model {bad!r} rejected", False)
    except ValueError:
        _c(f"invalid/missing model {bad!r} rejected", True)

print("\n--- renderer runs from rows alone, display cap active")
n = 900
X3 = rng.uniform(-200, 200, n); Y3 = rng.uniform(-200, 200, n); R3 = np.hypot(X3, Y3)
O3 = (rng.uniform(size=n) < np.clip(1 - R3 / 150, 0, 1)).astype(float)
pdf = CE.render_outcome_maps(os.path.join(tmp, "m.pdf"), "activation", O3, elec=E6, sign=S6,
                             center=cen, dip_c=dc, dip_d=dd, span=200.0, bin_um=8.0,
                             edges=np.arange(0, 208, 8.0), i0=50.0, n_pulses=36, cfg=None,
                             X=X3, Y=Y3, CULT=np.zeros(n), DN=R3, RD=R3, TP=rng.uniform(0, 180, n),
                             max_disks=50)
_c("PDF written and non-trivial", os.path.getsize(pdf) > 5000)

print("\nAll smoke tests passed.")
