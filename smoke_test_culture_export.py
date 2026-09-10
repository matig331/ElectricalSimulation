"""Offline smoke test for culture_export pure helpers (no NEURON)."""
import numpy as np
from culture_export import (electrode_center, dipole_axis_deg, dist_from_center,
                            dist_from_nearest_electrode, dipole_frame, directional_rt, smooth_P_field,
                            rel_orientation_deg, radial_bin_P, per_culture_P, logistic_decreasing,
                            fit_logistic_decreasing, assign_morphologies,
                            grid_P_2d, iso_radii, activation_area_per_culture, per_soma_P)

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

print("\nAll smoke tests passed.")
