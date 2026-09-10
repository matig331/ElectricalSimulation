"""
config.py -- single source of truth for the whole pipeline.

Edit HERE, then run the stages in pipeline.py. The one knob you change most is
N_NEURONS (few locally, the HPC value derived from density x area).
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class WellConfig:
    # --- morphologies (specimen tags found in the Eyal bundle) -------------
    morphologies: List[str] = field(default_factory=lambda: ["60308", "130303", "60303", "60311", "130305", "130306"])
    model: str = "rich"                 # "rich" (human L5) or "eyal" (na/kv)

    # --- electrodes (3Brain HyperCAM / CorePlate) --------------------------
    pitch_um: float = 60.0
    electrode_um: float = 25.0          # -> rmin = electrode_um/2
    bipolar: bool = True                # 2 anodes (+) / 2 cathodes (-)
    # Real device electrode geometry override: list of (x_um, y_um, sign); sign +1 anode / -1 cathode.
    # None -> generated 6-electrode 3+3 layout (field.default_array). Set to the lab's 6
    # activity-selected positions when known, e.g.
    #   electrodes = [(-60,30,1),(0,30,1),(60,30,1),(-60,-30,-1),(0,-30,-1),(60,-30,-1)]
    electrodes: Optional[Tuple] = None
    sigma_Sm: float = 1.5

    # --- well area (DERIVE with rich_footprint.py, then set here) ----------
    # area_half_um = the MINIMUM area half-width: the MEASURED activation footprint (the region a
    # soma is directly activated in), from the `area` stage. It is a LOWER BOUND -- the sampling must
    # cover at least this -- NOT the size of the network. The network/well is chosen separately below
    # (network_half_um). Already inferred. 180 is half. 

    area_half_um: float = 180.0 #dont change neither use 
    density_per_mm2: float = 1700.0     # biological density (HPC user may change to their culture's)
    # area_scan_half_um = how far the `area` stage SAMPLES to find the footprint (must be larger
    # than the true footprint or it clips). With 6 electrodes in a column the region is taller,
    # so scan wider (~200).
    area_scan_half_um: float = 200.0 #dont change neither use 
    # network_half_um = the area the HPC user actually simulates (the well, of order mm). None ->
    # fall back to area_half_um. This is what sets the biological neuron count on HPC.
    network_half_um: Optional[float] = 500

    # --- neuron count ------------------------------------------------------
    # n_neurons = somata per culture/well. THE 12 IS ARBITRARY, just for fast local tests.
    # On HPC set use_hpc_count=True and the count becomes biological automatically:
    #   N = density_per_mm2 x (2 * area / 1000)^2 ,  area = network_half_um (or area_half_um).
    n_neurons: int = 12
    use_hpc_count: bool = True         # HPC: True -> n_neurons_effective() = n_neurons_hpc()
    h_soma_um: float = 10.0
    seed: int = 0

    # --- lambda rule -------------------------------------------------------
    lambda_f_hz: float = 100.0
    axon_len_um: float = 0.0            # 0 = keep template (~6mm); >0 = shorten AIS/axon to this
    k_soma_mult: float = 1.0           # scale somatic/apical SKv3_1 (repolarizing K) -- raise to fix depol block
    ais_na_mult: float = 1.0           # scale AIS Na density
    ais_kv_mult: float = 1.0           # scale AIS Kv density
    na_soma_mult: float = 1.0          # scale somatic/apical NaTa_t (lower to tame firing rate)

    # --- slicing (3D -> 2D complexities) -----------------------------------
    layers_um: Tuple[float, ...] = (40.0, 80.0, 120.0)
    slice_thresh: float = 0.80

    # --- stimulation (lab params) + ramp (no Dirac) ------------------------
    i0_uA: float = 50.0
    phase_dur_ms: float = 0.25          # per phase (500 us total)
    anodic_first: bool = True
    ramp_us: float = 100.0              # trapezoid edges -> bounded dVe/dt
    interphase_us: float = 0.0
    n_pulses: int = 1                   # 1 = single pulse; lab train: 36 (0.2 Hz x 3 min)
    stim_freq_hz: float = 0.2           # 0.2 Hz -> ISI 5000 ms
    v_rest_mV: float = -73.9            # settled rest (step B); correct v_init
    baseline_ms: float = 100.0
    post_ms: float = 150.0
    dt_ms: float = 0.025
    settle_tol_mV_per_ms: float = 1e-2

    # --- SAMPLING: ONE GLOBAL KNOB for every stage --------------------------
    # n_samples = number of RANDOM soma positions drawn per morphology x layer.
    # EVERY sampling stage (recruitment, recruitment_export, footprint/area, bda)
    # uses it via random_positions() below. Each position is tested over `orient_deg`
    # orientations. Change ONE number here and all stages follow.
    #   cost of a sampling stage ~ n_samples * len(orient_deg) * n_morphologies * n_layers
    n_samples: int = 40
    orient_deg: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

    # --- CULTURE-based sampling (per the reviewer's spec) --------------------
    # A "culture" = one random realisation: ALL morphologies present, each neuron a
    # random soma position + random orientation. The SAME placement is re-simulated at
    # each layer thickness (3 sims per culture). See culture_export.py.
    # ONE dial: n_cultures (more cultures = more statistics, linear cost). The number of
    # neurons in a culture is the biological well count = n_neurons (above), NOT a separate knob.
    n_cultures: int = 384            # number of independent random cultures (THE dial)
    # distance metric for the P(distance) curve & area:
    #   "centroid" -- relative distance from the array centre (Sholl-like gradient)
    #   "nearest"  -- distance to the nearest electrode (cleanest for spread electrodes)
    #   "dipole"   -- 3D distance from the central-dipole centre, with the soma's angle to the
    #                dipole direction (- -> +) recorded (directional/vectorial). See culture_export.
    culture_distance_mode: str = "dipole"

    # --- stimulation DURATION (statistics over >= 1 min) --------------------
    # The lab delivers 3 min at 0.2 Hz. For the per-culture statistic the reviewer wants
    # >= 1 min delivered. Duration -> number of pulses via the frequency.
    stim_duration_s: float = 80.0  # total stimulation epoch (s); lab = 180 s (3 min)

    # --- before/during/after sweep (bda) -----------------------------------
    # kept for reference; bda now also uses n_samples random positions (orient random per pos).
    bda_n_positions: int = 3      # bda is a VISUAL check: a few positions x 3 layers (e.g. 3 -> 9 pages)
    bda_n_rot: int = 1            # one random orientation per position (visual only)

    @property
    def rmin_um(self) -> float:
        return self.electrode_um / 2.0

    def random_positions(self, n=None, rng=None, half=None):
        """THE shared sampler used by every stage. Returns an (n, 2) array of random
        (x, y) soma positions, uniform in the +/-`half` box. `n` defaults to n_samples;
        `half` defaults to area_half_um (a stage may pass its own footprint span instead).
        Seeded from `seed` when no rng is given, so runs are reproducible."""
        n = self.n_samples if n is None else int(n)
        half = self.area_half_um if half is None else float(half)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        return rng.uniform(-half, half, size=(int(n), 2))

    @property
    def isi_ms(self) -> float:
        return 1000.0 / self.stim_freq_hz

    def n_pulses_for_duration(self, duration_s=None) -> int:
        """Number of pulses that fill a stimulation epoch of `duration_s` (default
        stim_duration_s) at stim_freq_hz. This is what the per-culture statistic is
        computed over (reviewer: >= 1 min). At 0.2 Hz, 60 s -> 12 pulses, 180 s -> 36."""
        d = self.stim_duration_s if duration_s is None else float(duration_s)
        return max(1, int(round(d * self.stim_freq_hz)))

    def n_neurons_hpc(self) -> int:
        """Biological soma count = density x area, where area = network_half_um if set (the well the
        HPC user simulates), else area_half_um (the measured footprint = minimum).
        area (mm^2) = (2 * half / 1000)^2 ;  N = density_per_mm2 * area.
        Examples at 1300/mm^2:  half=130 -> ~88 , 180 -> ~168 , 200 -> ~208 , 500 -> ~1300 , 1000 -> ~5200."""
        half = self.area_half_um if self.network_half_um is None else float(self.network_half_um)
        area_mm2 = (2 * half / 1000.0) ** 2
        return int(round(self.density_per_mm2 * area_mm2))

    def n_neurons_effective(self) -> int:
        """The count culture_export / phase_split actually use: n_neurons_hpc() when use_hpc_count
        is True (HPC), else the hand-set n_neurons (local test)."""
        return self.n_neurons_hpc() if self.use_hpc_count else int(self.n_neurons)

    def span_half_um(self) -> float:
        """Sampling half-extent for culture_export: the network area if chosen, else the measured
        footprint (never smaller than the footprint)."""
        base = self.area_half_um if self.network_half_um is None else float(self.network_half_um)
        return max(float(base), float(self.area_half_um))


CFG = WellConfig()
