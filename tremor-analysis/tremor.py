"""Quantify shaking from a joint position time series.

Frequency, not counting. Tremor is an oscillation, so its frequency and amplitude
come from the spectrum. That sidesteps the hard part of counting discrete
movements, which is fragile exactly when amplitude decrements.

THREE BANDS
  voluntary   0.2-2 Hz    reaching, opening a fist, walking
  tremor      3-8 Hz      Parkinsonian rest tremor sits at 4-6 Hz
  noise      12-25 Hz     above physiological tremor and below Nyquist at 60 fps,
                          so whatever lands here is measurement noise

The noise band is the useful trick: it gives a per-recording, per-joint noise
estimate without needing a separate baseline recording, so every tremor amplitude
can be reported with the SNR that earned it.

IN-PLANE VERSUS DEPTH — REPORT THEM SEPARATELY
Measured on this project's recordings, the noise floor in the tremor band is
~1.1 mm for the 2D image-plane track and ~11 mm along the depth axis: a factor of
ten. So depth cannot resolve a 1 cm tremor and the in-plane figure is the one to
trust. Point the camera so the shaking is across the frame, not toward the lens.
Mixing the two axes into a single 3D amplitude would let the noisy axis dominate,
which is why nothing here does that.
"""
import numpy as np

VOLUNTARY_BAND = (0.2, 2.0)
TREMOR_BAND = (3.0, 8.0)
NOISE_BAND = (12.0, 25.0)

# A joint is called tremulous in a window when its tremor-band amplitude exceeds
# this multiple of its own noise-band amplitude. 3x is a conventional detection
# threshold and is deliberately conservative.
SNR_THRESHOLD = 3.0

# Voluntary movement in the 0.2-2 Hz band is not sinusoidal, so it leaks
# harmonics into the tremor band. When the voluntary amplitude exceeds the
# tremor-band amplitude by more than this factor, the tremor reading is treated
# as contaminated rather than real. Tremor should be assessed on a recording of
# the limb AT REST, not during a movement task.
VOLUNTARY_DOMINANCE = 3.0

# Window for constancy. Long enough to resolve a 3 Hz cycle several times over,
# short enough that intermittent tremor is not averaged away.
CONSTANCY_WINDOW_S = 2.0

# RMS -> peak-to-peak for a sinusoid. Reported because clinical amplitude
# criteria are excursions, not RMS.
RMS_TO_PP = 2.0 * np.sqrt(2.0)


def _fill(v):
    """Linearly interpolate NaN gaps so the FFT is defined. Returns None if the
    series is too sparse to mean anything."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < max(64, int(0.5 * v.size)):
        return None
    idx = np.arange(v.size)
    return np.interp(idx, idx[ok], v[ok])


def band_rms(v, fps, lo, hi):
    """RMS of the signal restricted to [lo, hi] Hz.

    The Hann window is compensated by sqrt(8/3) so the result is comparable to an
    unwindowed RMS rather than being biased low.
    """
    v = _fill(v)
    if v is None:
        return np.nan
    v = v - v.mean()
    n = v.size
    F = np.fft.rfft(v * np.hanning(n))
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    F[(f < lo) | (f > hi)] = 0
    return float(np.sqrt(np.mean(np.fft.irfft(F, n=n) ** 2)) * np.sqrt(8.0 / 3.0))


def dominant_frequency(v, fps, band=TREMOR_BAND):
    """Peak frequency inside the band, and how concentrated the peak is.

    sharpness is the fraction of in-band power sitting within +/-0.5 Hz of the
    peak: high for a genuine oscillation, low for broadband noise that happens to
    have a maximum somewhere.
    """
    v = _fill(v)
    if v is None:
        return np.nan, np.nan
    v = v - v.mean()
    n = v.size
    P = np.abs(np.fft.rfft(v * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or P[m].sum() <= 0:
        return np.nan, np.nan
    peak = float(f[m][np.argmax(P[m])])
    near = m & (np.abs(f - peak) <= 0.5)
    return peak, float(P[near].sum() / P[m].sum())


def frequency_resolution(n_samples, fps):
    """Hz per FFT bin — how precisely a frequency can be pinned down."""
    return fps / float(n_samples) if n_samples else np.nan


def constancy(v, fps, band=TREMOR_BAND, noise_band=NOISE_BAND,
              window_s=CONSTANCY_WINDOW_S, snr=SNR_THRESHOLD):
    """Fraction of time the joint is tremulous, 0-1.

    Slides a window, and calls each one tremulous when its tremor-band amplitude
    beats its noise-band amplitude by `snr`. Returns (fraction, n_windows).
    """
    v = _fill(v)
    if v is None:
        return np.nan, 0
    w = int(round(window_s * fps))
    if w < 32 or v.size < w:
        return np.nan, 0
    step = max(1, w // 2)
    hits = tot = 0
    for s in range(0, v.size - w + 1, step):
        seg = v[s:s + w]
        a = band_rms(seg, fps, *band)
        nz = band_rms(seg, fps, *noise_band)
        if np.isfinite(a) and np.isfinite(nz) and nz > 0:
            tot += 1
            if a / nz >= snr:
                hits += 1
    return (hits / tot if tot else np.nan), tot


def analyse_axis(series, fps, label):
    """Full tremor description of one scalar displacement series, in mm."""
    amp = band_rms(series, fps, *TREMOR_BAND)
    noise = band_rms(series, fps, *NOISE_BAND)
    vol = band_rms(series, fps, *VOLUNTARY_BAND)
    freq, sharp = dominant_frequency(series, fps)
    frac, nwin = constancy(series, fps)

    snr = (amp / noise) if (np.isfinite(amp) and np.isfinite(noise)
                            and noise > 0) else np.nan
    # Voluntary movement is not a pure sinusoid, so a vigorous 1-2 Hz task throws
    # harmonics straight into the 3-8 Hz band and beats the SNR test on noise
    # alone. Verified on a healthy open/close recording, where nearly every finger
    # joint looked tremulous at 3-4 Hz with amplitude growing toward the tips —
    # the signature of finger flexion, not tremor.
    vol_ratio = (vol / amp) if (np.isfinite(vol) and np.isfinite(amp)
                                and amp > 0) else np.nan
    contaminated = bool(np.isfinite(vol_ratio)
                        and vol_ratio > VOLUNTARY_DOMINANCE)
    return {
        "axis": label,
        "amp_rms_mm": amp,
        "amp_pp_mm": amp * RMS_TO_PP if np.isfinite(amp) else np.nan,
        "noise_rms_mm": noise,
        "snr": snr,
        # a joint only counts as tremulous if it clears the noise AND is not
        # swamped by voluntary movement
        "detected": bool(np.isfinite(snr) and snr >= SNR_THRESHOLD
                         and not contaminated),
        "contaminated": contaminated,
        "voluntary_ratio": vol_ratio,
        "dom_freq_hz": freq,
        "peak_sharpness": sharp,
        "constancy": frac,
        "n_windows": nwin,
        "voluntary_rms_mm": vol,
    }


def analyse_joint(px_mm, cam_mm, fps):
    """Tremor for one joint from both sources.

    px_mm  (T, 2)  image-plane track converted to mm — the trustworthy one
    cam_mm (T, 3)  absolute 3D track in mm; its z axis is ~10x noisier

    In-plane magnitude is taken as the RMS combination of the two in-plane axes,
    which is correct for independent components.
    """
    out = {}
    if px_mm is not None:
        ax = band_rms(px_mm[:, 0], fps, *TREMOR_BAND)
        ay = band_rms(px_mm[:, 1], fps, *TREMOR_BAND)
        mag = np.hypot(ax, ay)
        # describe the axis carrying more of the oscillation, then override the
        # amplitude with the combined in-plane magnitude
        base = analyse_axis(px_mm[:, 0] if ax >= ay else px_mm[:, 1], fps,
                            "2D in-plane")
        base["amp_rms_mm"] = mag
        base["amp_pp_mm"] = mag * RMS_TO_PP
        nz = base["noise_rms_mm"]
        base["snr"] = mag / nz if (np.isfinite(nz) and nz > 0) else np.nan
        # the contamination check has to be redone against the combined in-plane
        # magnitude, not the single axis analyse_axis happened to look at
        vol = base["voluntary_rms_mm"]
        base["voluntary_ratio"] = (vol / mag) if (np.isfinite(vol) and mag > 0) \
            else np.nan
        base["contaminated"] = bool(np.isfinite(base["voluntary_ratio"])
                                    and base["voluntary_ratio"] > VOLUNTARY_DOMINANCE)
        base["detected"] = bool(np.isfinite(base["snr"])
                                and base["snr"] >= SNR_THRESHOLD
                                and not base["contaminated"])
        out["in_plane_2d"] = base
    if cam_mm is not None:
        out["depth_3d"] = analyse_axis(cam_mm[:, 2], fps, "3D depth (noisy)")
    return out


def analyse_track(track, mm_per_px, reference=None):
    """Tremor for every joint in an extracted track.

    reference: index of a joint to subtract first. Referencing a fingertip to the
    wrist isolates finger tremor from whole-hand movement; leaving it None keeps
    the joint's absolute motion, which is what rest tremor of a limb means.
    Camera shake is indistinguishable from real tremor in the absolute case, so
    check `landmarks.camera_motion` before trusting it.
    """
    fps = track["geom"]["fps"]
    px = track["px"].copy()
    cam = track["cam"].copy() * 1000.0            # metres -> mm
    px_mm = px * mm_per_px

    if reference is not None:
        px_mm = px_mm - px_mm[:, reference:reference + 1, :]
        cam = cam - cam[:, reference:reference + 1, :]

    results = {}
    for i, name in enumerate(track["names"]):
        results[name] = analyse_joint(px_mm[:, i, :], cam[:, i, :], fps)
    return {"fps": fps, "reference": (track["names"][reference]
                                      if reference is not None else None),
            "n_frames": px.shape[0],
            "freq_resolution_hz": frequency_resolution(px.shape[0], fps),
            "joints": results}
