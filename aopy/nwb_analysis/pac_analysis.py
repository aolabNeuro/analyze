"""
pac_nwb_cache.py
================
Utilities for caching Hilbert-transform phase and amplitude envelopes
(from bandpass-filtered LFP) inside an existing NWB file, using
pynwb's DecompositionSeries.

Design decisions
----------------
- Phase (np.angle) and amplitude (np.abs) are stored as separate
  DecompositionSeries inside processing/ecephys.
- All bands in one call are packed into a single 3D array
  (time × channels × bands) per metric, matching the DecompositionSeries spec.
- Versioning: each unique (band_set, fs_target) combination gets its own
  named entry, so multiple parameterisations coexist in the same file.
- Downsampling: uses nap.TsdTensor.decimate() so timestamps are carried
  through by pynapple rather than recomputed manually. The factor must be
  an integer — for a 1000 Hz native rate, valid targets include 500, 250,
  200, 125, 100 Hz. Default is 2 × f_high_max (Nyquist of highest band).
- Overwrite: append by default. Pass overwrite='<dataset_name>' to delete
  and replace a specific existing entry by name.
- Loading: returns nap.TsdTensor objects (n_time × n_channels × n_bands)
  so .restrict(), slicing by band/channel, and all other pynapple methods
  work directly without any intermediate construction.

Entry naming convention
-----------------------
  hilbert_phase__{tag}__{fs_target}Hz
  hilbert_amplitude__{tag}__{fs_target}Hz

where <tag> is built from the band set:
  delta0.1-4__theta4-12__beta14-30__hg80-150

Usage
-----
  from pac_nwb_cache import cache_hilbert, load_hilbert, list_cached_hilbert

  # --- compute and cache ---
  bands      = [(0.1, 4), (4, 12), (14, 30), (80, 150)]
  band_names = ['delta', 'theta', 'beta', 'hg']

  cache_hilbert(
      nwb_path   = 'beignet_5927.nwb',
      lfp_tsd    = lfp_frame,          # pynapple TsdFrame, full session
      bands      = bands,
      band_names = band_names,
      fs_target  = None,               # default: 2 * max(f_hi) = 300 Hz
      overwrite  = None,               # or e.g. 'hilbert_phase__...__300Hz'
  )

  # --- load back as TsdTensors ---
  phase, amp = load_hilbert('beignet_5927.nwb', bands, band_names)
  # phase: TsdTensor (n_time, n_channels, n_bands) — timestamps included

  # Restrict to a trial epoch:
  epoch = nap.IntervalSet(start=trials['go_cue_time'], end=trials['go_cue_time'] + 0.5)
  phase_epoch = phase.restrict(epoch)

  # Select a band by index → TsdFrame (n_time, n_channels):
  theta_idx = band_names.index('theta')
  phase_theta = phase_epoch[:, :, theta_idx]

  # --- inspect what is stored ---
  list_cached_hilbert('beignet_5927.nwb')
"""

from __future__ import annotations

import json
import warnings
from math import gcd
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

import pynapple as nap
from pynwb import NWBHDF5IO
from pynwb.misc import DecompositionSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_tag(bands: list[tuple[float, float]], band_names: list[str]) -> str:
    """
    Build a compact, filesystem-safe string that uniquely identifies a band set.

    Example:
        bands      = [(0.1, 4), (4, 12), (14, 30), (80, 150)]
        band_names = ['delta', 'theta', 'beta', 'hg']
        → 'delta0.1-4__theta4-12__beta14-30__hg80-150'
    """
    parts = [f"{name}{lo}-{hi}" for (lo, hi), name in zip(bands, band_names)]
    return "__".join(parts)


def _entry_names(tag: str, fs_target: int) -> tuple[str, str]:
    """Return (phase_name, amplitude_name) for a given tag and sample rate."""
    suffix = f"{fs_target}Hz"
    return (
        f"hilbert_phase__{tag}__{suffix}",
        f"hilbert_amplitude__{tag}__{suffix}",
    )


def _valid_decimate_targets(fs_native: int) -> list[int]:
    """
    Return all integer rates that are exact integer decimations of fs_native,
    in ascending order. These are the rates r where fs_native / r is an integer.

    Example: fs_native=1000 → [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100,
                                125, 200, 250, 500, 1000]
    """
    return sorted([fs_native // d for d in range(1, fs_native + 1)
                   if fs_native % d == 0])


def _resolve_fs_target(bands: list[tuple[float, float]], fs_native: int,
                       fs_target: Optional[int]) -> int:
    """
    Resolve the target sample rate, guaranteeing it is an exact integer
    decimation of fs_native (required by nap.TsdTensor.decimate()).

    If fs_target is None, selects the lowest rate that is still above the
    Nyquist limit of the highest band (2 × f_high_max) and is an exact
    integer divisor of fs_native.

    If fs_target is explicitly provided but is not an exact divisor, snaps
    up to the nearest valid rate and prints a warning.

    Raises ValueError if no valid rate exists above Nyquist (which would
    only happen if fs_native itself is below 2 × f_high_max).
    """
    f_high_max = max(hi for _, hi in bands)
    nyquist_min = 2 * f_high_max          # minimum acceptable rate
    valid = _valid_decimate_targets(fs_native)

    if fs_target is None:
        # Pick the smallest valid rate that clears Nyquist
        candidates = [r for r in valid if r > nyquist_min]
        if not candidates:
            raise ValueError(
                f"fs_native ({fs_native} Hz) is at or below the Nyquist minimum "
                f"({nyquist_min} Hz) for the highest band. Cannot downsample."
            )
        chosen = candidates[0]
        if chosen != fs_native:
            print(f"  [fs_target] Auto-selected {chosen} Hz "
                  f"(lowest exact decimation of {fs_native} Hz above Nyquist {nyquist_min} Hz)")
        return chosen
    else:
        if fs_target in valid:
            return fs_target
        # Snap up to nearest valid rate above the requested target
        candidates = [r for r in valid if r >= fs_target]
        if not candidates:
            return fs_native   # requested target >= native rate, no downsampling
        snapped = candidates[0]
        print(
            f"  [fs_target] Requested {fs_target} Hz is not an exact integer decimation "
            f"of {fs_native} Hz. Snapping up to {snapped} Hz. "
            f"Valid targets for {fs_native} Hz native: {valid}"
        )
        return snapped


def _decimate_factor(fs_native: int, fs_target: int) -> int:
    """
    Return the integer decimation factor for nap.TsdTensor.decimate().
    _resolve_fs_target guarantees fs_target is an exact divisor of fs_native,
    so this is always a clean integer division.
    """
    return fs_native // fs_target


def _get_or_create_ecephys_module(nwb):
    """Return the 'ecephys' ProcessingModule, creating it if absent."""
    if "ecephys" in nwb.processing:
        return nwb.processing["ecephys"]
    return nwb.create_processing_module(
        name="ecephys",
        description="Processed electrophysiology data (bandpass, Hilbert, PAC)"
    )


def _remove_entry(nwb, name: str):
    """
    Remove a DecompositionSeries from processing/ecephys by name.

    pynwb does not expose a remove() method, so we reach into the underlying
    HDF5 file via the IO object's HDF5 handle. This is the standard workaround.
    Note: call this BEFORE io.write(), while the file is open in 'r+' mode.
    """
    mod = nwb.processing.get("ecephys")
    if mod is None or name not in mod.data_interfaces:
        warnings.warn(f"Entry '{name}' not found — nothing to remove.")
        return
    # Remove from the in-memory NWB object
    del mod.data_interfaces[name]


def _filter_hilbert_band(
    lfp_tsd: nap.TsdFrame,
    band: tuple[float, float],
    name: str,
    b_idx: int,
    fs: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Worker function for parallel band processing.

    Performs bandpass filter + Hilbert transform for a single band and returns
    phase and amplitude as float32 arrays. Defined at module level (not as a
    lambda or nested function) so joblib can pickle it for multiprocessing.

    Returns
    -------
    (b_idx, phase, amplitude) where phase and amplitude are
    (n_timepoints, n_channels) float32 arrays.
    """
    filtered = nap.apply_bandpass_filter(lfp_tsd, band, fs=fs)
    analytic = scipy_signal.hilbert(filtered.values, axis=0)
    return b_idx, np.angle(analytic).astype(np.float32), np.abs(analytic).astype(np.float32)


def _check_nwb_writable(nwb_path: str) -> None:
    """
    Raise a clear, early error if the NWB file cannot be opened for writing.

    Rather than trying to introspect open handles (which is unreliable — e.g.
    nap.load_file() may hold a handle invisible to h5py.h5f.get_obj_ids()),
    this function simply attempts to open the file in 'r+' mode and immediately
    closes it again. If that fails, HDF5 raises its own OSError which we catch
    and re-raise with a more informative message before any computation starts.

    Raises
    ------
    IOError
        If the file does not exist.
    OSError
        If the file is held open by another handle (e.g. nap.load_file(),
        a lingering NWBHDF5IO context, or a previous failed write), with
        a message explaining the likely cause and how to fix it.
    """
    import os
    import h5py

    nwb_path = str(nwb_path)

    if not os.path.exists(nwb_path):
        raise IOError(f"NWB file not found: {nwb_path}")

    try:
        # Attempt a real r+ open and close immediately — the only reliable
        # way to detect any conflicting handle regardless of its origin.
        f = h5py.File(nwb_path, "r+")
        f.close()
    except OSError as e:
        raise OSError(
            f"\n\nCannot open NWB file for writing:\n"
            f"  {nwb_path}\n\n"
            f"The file is likely already open in read-only mode in this session.\n"
            f"HDF5 does not allow a second open in write mode while any read handle\n"
            f"exists, regardless of where it was opened.\n\n"
            f"Common causes and fixes:\n"
            f"  data = nap.load_file(path)   -> nap.load_file() holds the file open\n"
            f"                                  for the lifetime of the returned object.\n"
            f"                                  Call data.close() before calling\n"
            f"                                  cache_hilbert(), then reload afterward:\n"
            f"                                      data.close()\n"
            f"                                      cache_hilbert(...)\n"
            f"                                      data = nap.load_file(path)\n"
            f"  NWBHDF5IO(path, 'r')         -> always use as a context manager so it\n"
            f"                                  closes automatically on exit.\n"
            f"  A previous cache_hilbert() that raised mid-write -> restart the kernel.\n\n"
            f"Original error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_cached_hilbert(nwb_path: str) -> list[dict]:
    """
    Print and return metadata for all cached Hilbert entries in the NWB file.

    Returns a list of dicts with keys: name, metric, bands, fs_target_hz, shape.
    """
    entries = []
    with NWBHDF5IO(nwb_path, "r") as io:
        nwb = io.read()
        if "ecephys" not in nwb.processing:
            print("No 'ecephys' processing module found.")
            return entries

        mod = nwb.processing["ecephys"]
        for name, obj in mod.data_interfaces.items():
            if not name.startswith("hilbert_"):
                continue
            meta = {
                "name": name,
                "metric": obj.metric if hasattr(obj, "metric") else "unknown",
                "shape": obj.data.shape if hasattr(obj.data, "shape") else "?",
                "description": obj.description,
            }
            try:
                desc = json.loads(obj.description)
                meta["bands"] = desc.get("bands")
                meta["fs_target_hz"] = desc.get("fs_target_hz")
            except (json.JSONDecodeError, AttributeError):
                pass
            entries.append(meta)
            print(f"  {name}")
            print(f"    metric      : {meta['metric']}")
            print(f"    shape       : {meta['shape']}")
            if "bands" in meta:
                print(f"    bands       : {meta['bands']}")
            if "fs_target_hz" in meta:
                print(f"    fs_target   : {meta['fs_target_hz']} Hz")
            print()
    return entries


def cache_hilbert(
    nwb_path: str,
    lfp_tsd,                          # pynapple TsdFrame, full session
    bands: list[tuple[float, float]],
    band_names: list[str],
    fs_target: Optional[int] = None,  # None → 2 × f_high_max
    overwrite: Optional[str] = None,  # name of dataset to overwrite, or None
    compression_level: int = 4,       # gzip level 0-9
    source_series_name: str = "ElectricalSeriesRaw",
    n_jobs: int = 1,                  # parallel workers: 1=serial, -1=all cores
) -> tuple[str, str]:
    """
    Compute full-session bandpass + Hilbert for all bands, then write
    phase and amplitude DecompositionSeries into processing/ecephys.

    Parameters
    ----------
    nwb_path : str
        Path to the NWB file (opened in 'r+' mode).
    lfp_tsd : pynapple.TsdFrame
        Full-session LFP. Must have .rate (Hz) and .t (timestamps in seconds).
    bands : list of (f_lo, f_hi) tuples
        Frequency band edges in Hz.
    band_names : list of str
        One label per band. Must match len(bands).
    fs_target : int or None
        Target sample rate for stored data. None → 2 × max(f_hi).
        Pass the native rate explicitly to skip downsampling.
    overwrite : str or None
        Name of an existing dataset to delete before writing. The paired
        dataset (phase ↔ amplitude) is also removed automatically.
        Example: overwrite='hilbert_phase__delta0.1-4__theta4-12__300Hz'
    compression_level : int
        gzip compression level (0 = none, 9 = max). Default 4.
    source_series_name : str
        Name of the source ElectricalSeries in acquisition (for linking).
    n_jobs : int
        Number of parallel workers for the filter + Hilbert step.
        1 = serial (default), -1 = all available cores, any positive
        integer = that many workers. Uses joblib with prefer='threads'
        so results are shared directly in memory rather than serialized
        through a pipe — avoiding the MemoryError that occurs when
        large arrays are returned via multiprocessing. The GIL is
        released during scipy/numpy C extensions so real parallelism
        is still achieved. When batching across many sessions in an
        outer loop, prefer n_jobs=1 here and parallelise at the session
        level instead to avoid memory pressure.

    Returns
    -------
    (phase_name, amplitude_name) : the NWB entry names that were written.
    """
    assert len(bands) == len(band_names), "bands and band_names must have equal length"

    # ------------------------------------------------------------------
    # 0. Pre-flight checks — fail fast before any computation
    # ------------------------------------------------------------------

    # Check 1: lfp_tsd must be fully in memory (not HDF5-backed).
    # When loaded via nap.load_file(), TsdFrame data is a lazy h5py.Dataset
    # pointer rather than a numpy array. joblib's multiprocessing backend
    # pickles everything it sends to workers, and h5py objects cannot be
    # pickled — so parallel dispatch will fail with a PicklingError.
    # Even for n_jobs=1 the HDF5 handle creates subtle risks if the file
    # is closed mid-computation.
    # Fix: materialise the TsdFrame before calling this function:
    #   lfp_frame = lfp_frame.restrict(lfp_frame.time_support)
    import h5py as _h5py
    if isinstance(lfp_tsd.values, _h5py.Dataset):
        raise TypeError(
            "lfp_tsd appears to be lazily backed by an open HDF5 file "
            "(lfp_tsd.values is an h5py.Dataset, not a numpy array).\n\n"
            "Materialise it into memory before calling cache_hilbert():\n\n"
            "    lfp_frame = lfp_frame.restrict(lfp_frame.time_support)\n\n"
            "This is required for parallel processing (n_jobs != 1) and "
            "avoids file-handle conflicts during the write step."
        )

    # Check 2: file is writable (not held open by another handle)
    _check_nwb_writable(nwb_path)

    # Check 2: resolve names and verify the cache entry doesn't already exist
    #          (or that overwrite explicitly targets it), before spending time
    #          on filtering and Hilbert transforms.
    # Round to the nearest integer Hz to absorb floating point drift in
    # timestamps stored in NWB files (e.g. 1000.0004173... -> 1000).
    # The nominal rate is always a round number; the fractional part is
    # an artefact of how pynapple derives .rate from stored timestamps.
    fs_native = round(lfp_tsd.rate)
    fs_out = _resolve_fs_target(bands, fs_native, fs_target)
    tag = _build_tag(bands, band_names)
    phase_name, amp_name = _entry_names(tag, fs_out)

    with NWBHDF5IO(nwb_path, "r") as io:
        nwb = io.read()
        mod = nwb.processing.get("ecephys")
        existing = set(mod.data_interfaces.keys()) if mod is not None else set()

    if overwrite is None and (phase_name in existing or amp_name in existing):
        print(
            f"[cache_hilbert] Entries already exist and overwrite=None — skipping computation.\n"
            f"  phase : {phase_name}\n"
            f"  amp   : {amp_name}\n"
            f"  Pass overwrite='<name>' to replace a specific entry."
        )
        return phase_name, amp_name

    if overwrite is not None and overwrite not in existing:
        raise KeyError(
            f"overwrite='{overwrite}' was specified but no entry with that name exists.\n"
            f"Existing hilbert entries: {[k for k in existing if k.startswith('hilbert_')]}\n"
            f"Check list_cached_hilbert(nwb_path) for valid names."
        )

    print(f"[cache_hilbert] Band set tag : {tag}")
    print(f"[cache_hilbert] Native fs    : {fs_native} Hz → storing at {fs_out} Hz")
    print(f"[cache_hilbert] Phase entry  : {phase_name}")
    print(f"[cache_hilbert] Amp entry    : {amp_name}")

    # ------------------------------------------------------------------
    # 1. Compute bandpass + Hilbert for every band (full session)
    #
    # Each band is independent, so they can be processed in parallel.
    # _filter_hilbert_band() is a module-level function.
    # With n_jobs=1 this is identical to the serial loop; with n_jobs=-1
    # all bands run simultaneously. prefer='threads' means results are
    # shared in memory rather than serialized through a pipe — this avoids
    # the MemoryError that occurs with loky (multiprocessing) when phase
    # and amplitude arrays (~500MB–1GB per band) are returned from workers.
    # The GIL is released during scipy/numpy C extensions so threading
    # still achieves real parallelism for the compute-heavy parts.
    # ------------------------------------------------------------------
    from joblib import Parallel, delayed

    n_time, n_ch = lfp_tsd.shape
    n_bands = len(bands)

    phase_all = np.empty((n_time, n_ch, n_bands), dtype=np.float32)
    amp_all   = np.empty((n_time, n_ch, n_bands), dtype=np.float32)

    if n_jobs == 1:
        for b_idx, (band, name) in enumerate(zip(bands, band_names)):
            print(f"  Filtering + Hilbert: {name} {band} Hz ...")
            b_idx_out, phase, amp = _filter_hilbert_band(lfp_tsd, band, name, b_idx, fs_native)
            phase_all[:, :, b_idx_out] = phase
            amp_all[:, :, b_idx_out]   = amp
    else:
        print(f"  Filtering + Hilbert: {n_bands} bands in parallel (n_jobs={n_jobs}) ...")
        results = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
            delayed(_filter_hilbert_band)(lfp_tsd, band, name, b_idx, fs_native)
            for b_idx, (band, name) in enumerate(zip(bands, band_names))
        )
        for b_idx_out, phase, amp in results:
            phase_all[:, :, b_idx_out] = phase
            amp_all[:, :, b_idx_out]   = amp

    # ------------------------------------------------------------------
    # 2. Downsample using nap.TsdTensor.decimate()
    #
    # Wrapping as TsdTensor lets pynapple carry timestamps through the
    # decimation, eliminating the need to manually recompute a timestamp
    # array (np.linspace) that could drift from the actual sample positions.
    # decimate() applies an antialiasing filter before downsampling.
    # ------------------------------------------------------------------
    # Build TsdTensors — timestamps come from the original lfp_tsd
    phase_tensor = nap.TsdTensor(t=lfp_tsd.t, d=phase_all)
    amp_tensor   = nap.TsdTensor(t=lfp_tsd.t, d=amp_all)

    if fs_out < int(fs_native):
        print(f"  Downsampling to {fs_out} Hz via TsdTensor.decimate() ...")
        factor = _decimate_factor(fs_native, fs_out)
        phase_tensor = phase_tensor.decimate(factor)
        amp_tensor   = amp_tensor.decimate(factor)

    # Extract numpy arrays and timestamps from the (possibly decimated) tensors
    ph_ds = phase_tensor.values   # (n_time_ds, n_ch, n_bands)
    am_ds = amp_tensor.values
    ts_ds = phase_tensor.t        # timestamps are authoritative from pynapple

    print(f"  Stored shape: {ph_ds.shape}  (time × channels × bands)")

    # ------------------------------------------------------------------
    # 3. Build shared metadata
    # ------------------------------------------------------------------
    bands_meta = [
        {"band_name": name, "band_limits_hz": list(band)}
        for name, band in zip(band_names, bands)
    ]
    description = json.dumps({
        "source_series": source_series_name,
        "filter_method": "pynapple.apply_bandpass_filter",
        "hilbert_method": "scipy.signal.hilbert",
        "bands": bands_meta,
        "fs_native_hz": fs_native,
        "fs_target_hz": fs_out,
        "axes": ["time", "channel", "band"],
    })

    # bands DynamicTable rows (required by DecompositionSeries)
    band_table_data = {
        "band_name": band_names,
        "band_limits": [list(b) for b in bands],
    }

    # ------------------------------------------------------------------
    # 4. Write to NWB
    # ------------------------------------------------------------------
    with NWBHDF5IO(nwb_path, "r+") as io:
        nwb = io.read()
        mod = _get_or_create_ecephys_module(nwb)

        # Handle overwrite
        if overwrite is not None:
            # Infer the paired name
            if overwrite == phase_name:
                paired = amp_name
            elif overwrite == amp_name:
                paired = phase_name
            else:
                paired = None
                warnings.warn(
                    f"overwrite='{overwrite}' does not match the current entry names "
                    f"({phase_name}, {amp_name}). Only the named entry will be removed."
                )
            print(f"  Removing existing entry: {overwrite}")
            _remove_entry(nwb, overwrite)
            if paired and paired in mod.data_interfaces:
                print(f"  Removing paired entry : {paired}")
                _remove_entry(nwb, paired)

        # Try to link back to the source ElectricalSeries
        source_ts = nwb.acquisition.get(source_series_name)

        def _make_decomp(name, data, metric):
            kwargs = dict(
                name=name,
                description=description,
                metric=metric,
                data=H5DataIO(
                    data,
                    compression="gzip",
                    compression_opts=compression_level,
                ),
                timestamps=H5DataIO(
                    ts_ds.astype(np.float64),
                    compression="gzip",
                    compression_opts=compression_level,
                ),
            )
            if source_ts is not None:
                kwargs["source_timeseries"] = source_ts
            ds = DecompositionSeries(**kwargs)
            # Add band table
            for b_name, b_limits in zip(band_names, bands):
                ds.add_band(band_name=b_name, band_limits=b_limits)
            return ds

        phase_ds = _make_decomp(phase_name, ph_ds, "phase")
        amp_ds   = _make_decomp(amp_name,   am_ds, "amplitude")

        mod.add(phase_ds)
        mod.add(amp_ds)
        io.write(nwb)

    print(f"[cache_hilbert] Done. Wrote '{phase_name}' and '{amp_name}'.")
    return phase_name, amp_name


def load_hilbert(
    nwb_path: str,
    bands: list[tuple[float, float]],
    band_names: list[str],
    fs_target: Optional[int] = None,
) -> tuple[nap.TsdTensor, nap.TsdTensor]:
    """
    Load cached phase and amplitude from the NWB file as pynapple TsdTensors.

    The returned TsdTensors have shape (n_time, n_channels, n_bands) and carry
    timestamps natively, so all pynapple methods (.restrict(), slicing, etc.)
    work directly on them.

    Parameters
    ----------
    nwb_path : str
    bands : list of (f_lo, f_hi)
        Must match the bands used during cache_hilbert exactly.
    band_names : list of str
    fs_target : int or None
        Must match the fs_target used during cache_hilbert. If None, attempts
        to auto-match any cached entry for the given band set.

    Returns
    -------
    phase_tensor : nap.TsdTensor  shape (n_time, n_channels, n_bands)
    amp_tensor   : nap.TsdTensor  shape (n_time, n_channels, n_bands)

    The band axis (axis 2) corresponds to band_names in order.

    Examples
    --------
    phase, amp = load_hilbert(nwb_path, bands, band_names)

    # Restrict both tensors to a trial epoch in one call each:
    epoch = nap.IntervalSet(start=trials['go_cue_time'], end=trials['go_cue_time'] + 0.5)
    phase_epoch = phase.restrict(epoch)   # TsdTensor, same shape minus excluded time
    amp_epoch   = amp.restrict(epoch)

    # Select a specific band by index → returns a TsdFrame (channels as columns):
    theta_idx = band_names.index('theta')
    phase_theta = phase_epoch[:, :, theta_idx]   # TsdFrame (n_time, n_channels)

    # Or select a specific channel across all bands → TsdFrame (n_time, n_bands):
    phase_ch5 = phase_epoch[:, 5, :]

    # Pass to your existing PAC loop:
    MI_obs, MI_thresh, MI_pval = modulation_index_matrix_with_significance(
        phase_theta.restrict(tmp_epoch).values,   # (n_time, n_phase_chans)
        amp_epoch[:, :, hg_idx].values,
        n_surrogates=200,
    )
    """
    with NWBHDF5IO(nwb_path, "r") as io:
        nwb = io.read()

        if "ecephys" not in nwb.processing:
            raise KeyError("No 'ecephys' processing module found in this NWB file.")

        mod = nwb.processing["ecephys"]

        # Resolve entry names
        fs_native_guess = 1000.0  # fallback for name resolution only
        fs_out = _resolve_fs_target(bands, fs_native_guess, fs_target)
        tag = _build_tag(bands, band_names)
        phase_name, amp_name = _entry_names(tag, fs_out)

        # Auto-match if fs_target=None and exact name not found
        if fs_target is None and phase_name not in mod.data_interfaces:
            candidates = [k for k in mod.data_interfaces if f"hilbert_phase__{tag}__" in k]
            if candidates:
                phase_name = candidates[0]
                amp_name = phase_name.replace("hilbert_phase__", "hilbert_amplitude__")
                print(f"[load_hilbert] Auto-matched: {phase_name}")
            else:
                raise KeyError(
                    f"No cached entry found for band set '{tag}'. "
                    f"Available: {[k for k in mod.data_interfaces if k.startswith('hilbert_')]}"
                )

        if phase_name not in mod.data_interfaces:
            raise KeyError(
                f"Entry '{phase_name}' not found. "
                f"Run cache_hilbert() first, or check list_cached_hilbert()."
            )

        phase_ds = mod[phase_name]
        amp_ds   = mod[amp_name]

        # Load arrays: shape (n_time, n_channels, n_bands)
        phase_data = phase_ds.data[:]
        amp_data   = amp_ds.data[:]
        timestamps = phase_ds.timestamps[:]

    # Wrap as TsdTensors so all pynapple methods (restrict, slicing, etc.) are available.
    # Shape is (n_time, n_channels, n_bands) — band axis matches band_names order.
    phase_tensor = nap.TsdTensor(t=timestamps, d=phase_data.astype(np.float32))
    amp_tensor   = nap.TsdTensor(t=timestamps, d=amp_data.astype(np.float32))

    return phase_tensor, amp_tensor


def hilbert_cache_exists(
    nwb_path: str,
    bands: list[tuple[float, float]],
    band_names: list[str],
    fs_target: Optional[int] = None,
) -> bool:
    """
    Return True if a cached entry for this exact band set and fs_target exists.
    Useful for guarding recomputation in batch loops.

    Example
    -------
    if not hilbert_cache_exists(nwb_path, bands, band_names):
        cache_hilbert(nwb_path, lfp_frame, bands, band_names)
    """
    try:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            if "ecephys" not in nwb.processing:
                return False
            mod = nwb.processing["ecephys"]
            fs_native_guess = 1000.0
            fs_out = _resolve_fs_target(bands, fs_native_guess, fs_target)
            tag = _build_tag(bands, band_names)
            phase_name, _ = _entry_names(tag, fs_out)

            if phase_name in mod.data_interfaces:
                return True
            # Also check for auto-matched tag (any fs suffix)
            return any(f"hilbert_phase__{tag}__" in k for k in mod.data_interfaces)
    except Exception:
        return False