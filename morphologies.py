"""
morphologies.py -- find and prepare Eyal morphologies automatically.

Two jobs:
  * discovery: locate every specimen_*/morphology.asc inside the bundle
    (extracting eyal_hpc_bundle*.tar.gz on first use, like the old scripts),
    searching a few sensible roots so scripts run with no path argument.
  * preparation: build an axon-stripped copy of a morphology (for the
    with-/without-axon comparison), using morphio.
"""
import os
import glob
import tarfile
import tempfile

# roots searched when no explicit path is given (first hit wins)
_SEARCH_ROOTS = [
    os.path.dirname(os.path.abspath(__file__)),
    os.getcwd(),
    os.path.expanduser("~/Desktop/silico"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~"),
]


def _extract_bundle(tar_path):
    out = os.path.join(os.path.dirname(tar_path) or ".", "bundle_extracted")
    if not os.path.isdir(out):
        with tarfile.open(tar_path) as t:
            t.extractall(out)
    return out


def find_archive(roots=None):
    """Return the path to an `eyal_archive` directory, extracting the tarball
    if only the .tar.gz is present. Raises if nothing is found."""
    roots = roots or _SEARCH_ROOTS
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "eyal_archive"), recursive=True)
        if hits:
            return hits[0]
        tars = (glob.glob(os.path.join(root, "**", "eyal_hpc_bundle*.tar.gz"), recursive=True)
                + glob.glob(os.path.join(root, "**", "eyal_hpc_bundle*tar.gz"), recursive=True))
        if tars:
            ext = _extract_bundle(tars[0])
            hits = glob.glob(os.path.join(ext, "**", "eyal_archive"), recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(
        "Could not find eyal_archive or eyal_hpc_bundle*.tar.gz. "
        "Put the bundle next to these scripts, or pass an explicit path.")


def all_morphologies(archive=None):
    """Sorted list of (specimen_name, asc_path) for every morphology found."""
    archive = archive or find_archive()
    ascs = sorted(glob.glob(os.path.join(archive, "specimen_*", "morphology.asc")))
    return [(os.path.basename(os.path.dirname(a)), a) for a in ascs]


def find_one_morphology(preferred=None):
    """One morphology path -- for quick tests. `preferred` matches a substring
    of the specimen name (e.g. '60311')."""
    morphs = all_morphologies()
    if preferred:
        for name, path in morphs:
            if preferred in name:
                return path
    return morphs[0][1]


def prepared(asc_path, keep_axon=True, out_dir=None):
    """Write a morphio-sanitized copy and return its path.

    Removes unifurcations (single-child chains merged -> valid for writing) and,
    if keep_axon=False, deletes every axonal root section first. Both variants go
    through the SAME sanitization so a with-/without-axon comparison is fair.
    Written as .asc (preserves the soma representation; LFPy/NEURON load it).
    """
    import morphio
    import morphio.mut
    try:
        morphio.set_maximum_warnings(0)          # silence unifurcation-merge notes
    except Exception:
        pass
    m = morphio.mut.Morphology(asc_path)
    if not keep_axon:
        for s in [x for x in m.root_sections if x.type == morphio.SectionType.axon]:
            m.delete_section(s, recursive=True)
    m.remove_unifurcations()
    out_dir = out_dir or tempfile.gettempdir()
    name = os.path.basename(os.path.dirname(asc_path)) or "cell"
    out = os.path.join(out_dir, f"{name}_{'axon' if keep_axon else 'noaxon'}.asc")
    m.write(out)
    return out


def without_axon(asc_path, out_dir=None):
    """Back-compat shim: an axon-stripped, sanitized copy."""
    return prepared(asc_path, keep_axon=False, out_dir=out_dir)
