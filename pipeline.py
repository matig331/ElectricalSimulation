"""
pipeline.py -- one place to run every stage, all driven by config.py (CFG).

Edit config.py (especially n_neurons and area_half_um), then run the stages you
want below. FAST stages need no simulation; ACTIVE stages run the Rich model,
so keep the neuron/condition counts small locally.

Mapping of your checklist -> stage:
  4  lambda rule / compartments ....... stage_lambda      (FAST)
  5  Ve field distribution ............ stage_ve_field    (FAST)
  8  channel distribution ............. stage_channels    (FAST)
  7  no Dirac in the field ............ stage_waveform    (FAST)
  9  slicing 3D -> 2D, complexities ... stage_slicing     (FAST)
  1  correct area from activation ..... stage_area        (ACTIVE)
  -  homeostatic baseline (step B) .... stage_baseline    (ACTIVE)
  10 Vm/Ve before/during/after ........ stage_bda         (ACTIVE, few)
  3/11 well, N neurons, 3 morphs ...... stage_well        (FAST)
"""
from config import CFG


def stage_lambda(cfg=CFG):
    import lambda_metrics; lambda_metrics.main()                 # per morphology

def stage_ve_field(cfg=CFG):
    import ve_field; ve_field.main(i0_uA=cfg.i0_uA)

def stage_channels(cfg=CFG):
    import channel_atlas
    for m in cfg.morphologies:
        channel_atlas.main(pref=m, layers=cfg.layers_um)   # loops all layers internally

def stage_waveform(cfg=CFG):
    import numpy as np, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, field as F
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    t_on = cfg.baseline_ms; pad = 1.0; dur = 2 * cfg.phase_dur_ms
    # window CENTRED on the pulse (not 100 ms of baseline); fine dt to resolve the ramp
    t = np.arange(t_on - pad, t_on + dur + pad, cfg.dt_ms / 4)
    I = F.biphasic_current(t - t_on, cfg.i0_uA, cfg.phase_dur_ms,
                           cfg.anodic_first, cfg.ramp_us, cfg.interphase_us)
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(t, I*1e6); ax[0].set_ylabel("I (uA)")
    ax[0].set_title("biphasic current (ramped) -- pulse centred in the window")
    ax[1].plot(t, np.gradient(I, t*1e-3)); ax[1].set_ylabel("dI/dt (A/s)"); ax[1].set_xlabel("t (ms)")
    ax[1].set_title("bounded dI/dt -> no Dirac")
    for a in ax:
        a.axvspan(t_on, t_on + dur, color="0.9", zorder=0)   # shade the pulse
        a.set_xlim(t_on - pad, t_on + dur + pad)
    # zoom inset on the leading ramp (the 'step' the reviewer wants to see clearly)
    az = inset_axes(ax[0], width="42%", height="42%", loc="upper right")
    az.plot(t, I*1e6); az.axvspan(t_on, t_on + cfg.ramp_us/1000.0, color="0.85", zorder=0)
    az.set_xlim(t_on - cfg.ramp_us/1000.0*0.5, t_on + cfg.ramp_us/1000.0 + cfg.phase_dur_ms*0.4)
    az.set_title("zoom: leading ramp", fontsize=6); az.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig("waveform_check.pdf"); plt.close(fig)
    print("done: waveform_check.pdf")

def stage_slicing(cfg=CFG):
    import slicer
    morphs = [(f"specimen_{m}", __import__('morphologies').find_one_morphology(m))
              for m in cfg.morphologies]
    slicer.report_shapes(morphs=morphs, layers=cfg.layers_um, thr=cfg.slice_thresh)

def stage_area(cfg=CFG):
    # area study over ALL morphologies x layers -> combined envelope + margin.
    # Samples out to area_scan_half_um (wide enough for the 6-electrode column) so the
    # activation region is not clipped. Put the printed MEAN/ENVELOPE into config.area_half_um.
    import rich_footprint
    rich_footprint.multi(prefs=tuple(cfg.morphologies), layers=cfg.layers_um,
                         half=cfg.area_scan_half_um, i0_uA=cfg.i0_uA)

def stage_baseline(cfg=CFG):
    import baseline; baseline.main(pref=cfg.morphologies[0], layer=cfg.layers_um[-1])

def stage_bda(cfg=CFG):
    # Vm/Ve before/during/after at RANDOM positions x orientations, per layer,
    # for ALL morphologies. Counts are all set in config.py.
    import numpy as np, rich_stim
    rng = np.random.default_rng(cfg.seed)
    positions = [tuple(p) for p in rng.uniform(-cfg.area_half_um, cfg.area_half_um,
                                               size=(cfg.bda_n_positions, 2))]
    print(f"stage_bda: {len(cfg.morphologies)} morph x {len(cfg.layers_um)} layer x "
          f"{cfg.bda_n_positions} pos x {cfg.bda_n_rot} rot = "
          f"{len(cfg.morphologies)*len(cfg.layers_um)*cfg.bda_n_positions*cfg.bda_n_rot} sims")
    for m in cfg.morphologies:
        rich_stim.batch(pref=m, layers=cfg.layers_um,
                        positions=positions, n_rot=cfg.bda_n_rot,
                        baseline_ms=cfg.baseline_ms, post_ms=cfg.post_ms,
                        out_pdf=f"rich_stim_{m}.pdf")

def stage_area_check(cfg=CFG):
    import well; well.check_area(cfg)

def stage_well_active(cfg=CFG):
    # 3 morphologies in ONE well, ACTIVE, few neurons, Vm before/during/after
    import well; well.well_active(cfg)

def stage_channel_dynamics(cfg=CFG):
    import channel_dynamics; channel_dynamics.main()

def stage_recruitment(cfg=CFG):
    import recruitment; recruitment.main()

def stage_intrinsic(cfg=CFG):
    import intrinsic; intrinsic.main()

def stage_ais_scan(cfg=CFG):
    import intrinsic; intrinsic.ais_scan()

def stage_somatic_k_scan(cfg=CFG):
    import intrinsic; intrinsic.somatic_k_scan()

def stage_excitability_scan(cfg=CFG):
    import intrinsic; intrinsic.excitability_scan()

def stage_model_diagnostic(cfg=CFG):
    import intrinsic; intrinsic.model_diagnostic()

def stage_check_polarization(cfg=CFG):
    import coupling_checks; coupling_checks.check_polarization()

def stage_check_current_distance_k(cfg=CFG):
    import coupling_checks; coupling_checks.check_current_distance_k()

def stage_check_activating_function(cfg=CFG):
    import coupling_checks; coupling_checks.check_activating_function()

def stage_axon_sensitivity(cfg=CFG):
    import axon_sensitivity; axon_sensitivity.main()

def stage_subthreshold(cfg=CFG):
    import subthreshold; subthreshold.main()

def stage_well(cfg=CFG):
    import well; well.draw(cfg)

def stage_checks(cfg=CFG, active=True):
    import checks; checks.run_all(cfg, active=active)


def run_all(cfg=CFG):
    """One HPC entry point: run EVERY stage, all layers, all morphologies, with the
    counts set in config.py. Run stage_area FIRST (separately) to fix area_half_um,
    then call this. Heavy -- meant for HPC."""
    stage_checks(cfg, active=True)
    stage_waveform(cfg); stage_lambda(cfg); stage_ve_field(cfg)
    stage_channels(cfg); stage_slicing(cfg); stage_well(cfg)
    stage_area_check(cfg)
    stage_bda(cfg); stage_well_active(cfg); stage_channel_dynamics(cfg)
    stage_intrinsic(cfg); stage_recruitment(cfg); stage_subthreshold(cfg)
    print("run_all: done")


def _cli():
    import sys
    if len(sys.argv) > 1:
        name = sys.argv[1]
        fn = globals().get(name) or globals().get("stage_" + name)
        if callable(fn):
            fn(); return True
        print("unknown stage:", name)
        print("available:", ", ".join(sorted(n[6:] for n in globals() if n.startswith("stage_"))) + ", run_all")
        return True
    return False


if __name__ == "__main__":
    if not _cli():
        # no stage name given -> quick default demo
        stage_checks(active=False)
        stage_waveform()
        stage_well()
