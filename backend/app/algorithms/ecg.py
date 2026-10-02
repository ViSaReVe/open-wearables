"""Single-lead ECG R-peak detection and beat-to-beat HRV.

Pan-Tompkins style pipeline (bandpass -> derivative -> square -> moving-window integration ->
adaptive thresholding), implemented with numpy only. Every parameter is defined in physical
units (seconds / Hz) and converted with the sampling rate at runtime, because the sampling rate
of provider waveforms is not always documented and is estimated from the sample timestamps.

Provider-agnostic: callers translate vendor quality flags into boolean sample masks.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# QRS energy is concentrated in ~5-15 Hz. Below: baseline wander/respiration (<0.5 Hz) and
# P/T waves (mostly <5 Hz). Above: EMG and 50/60 Hz mains.
QRS_BAND_HZ: tuple[float, float] = (5.0, 15.0)
# Hamming-window FIR transition width is ~3.3 * fs / N Hz; 3 Hz at the 5 Hz edge -> N ~ 1.1 * fs.
FIR_TRANSITION_HZ = 3.0
# The 15 Hz upper band edge needs comfortable margin below Nyquist.
MIN_SAMPLING_RATE_HZ = 50.0
# ~ widest plausible QRS (normal 80-100 ms, up to ~120 ms with bundle-branch block). Shorter
# windows split one QRS into several peaks; longer ones merge the T wave in.
INTEGRATION_WINDOW_S = 0.150
# 60 / 240 bpm. Plausible spot-check HR range is ~30-240 bpm; the physiological absolute
# refractory period is ~200 ms.
REFRACTORY_S = 0.250
# A candidate this close to the previous beat with less than half its QRS slope is a T wave.
T_WAVE_WINDOW_S = 0.360
T_WAVE_SLOPE_RATIO = 0.5
# Pan-Tompkins searchback: no beat for 1.66 x recent mean RR -> re-search at half threshold.
SEARCHBACK_RR_FACTOR = 1.66
SEARCHBACK_RR_HISTORY = 8
THRESHOLD_INIT_S = 2.0
# R-peak refinement window around the integrator peak.
REFINE_HALF_WINDOW_S = 0.075
# Segments shorter than this are skipped: ~2 s threshold warm-up plus >= 3 beats. NOTE: at very
# low heart rates (< ~40 bpm) 5 s may hold fewer than 3 beats; such segments yield no RMSSD pairs.
MIN_SEGMENT_S = 5.0
# Physiological RR bounds (240 bpm .. 30 bpm).
RR_MIN_MS = 250.0
RR_MAX_MS = 2000.0
# Ectopic/artifact rule: reject RR deviating more than 20% from the local median.
ECTOPIC_TOLERANCE = 0.20
ECTOPIC_MEDIAN_WINDOW = 5
# Above this fraction of rejected intervals the beat train is not trusted for RMSSD: on synthetic
# data, detector precision collapses together with a rising artifact rate (see PROTOTYPE_NOTES.md).
MAX_ARTIFACT_FRACTION = 0.20
# Missing samples are bridged by linear interpolation when the hole is at most this long (about
# the half-width of an R wave, so a hole cannot hide a whole R upstroke); longer holes split the
# recording into separate segments.
MAX_BRIDGED_GAP_S = 0.025


def estimate_sampling_period_ms(
    times_ms: np.ndarray,
    segments: Sequence[slice] | None = None,
    sample_index: np.ndarray | None = None,
) -> float:
    """Estimate the sampling period from (possibly integer-rounded) sample timestamps.

    Least-squares slope of time vs. sample index, pooled across segments (common slope, separate
    intercepts). Rounding each timestamp to the nearest ms (e.g. 7/8 ms steps at 130 Hz) averages
    out in the fit, unlike taking the median of first differences. `sample_index` gives each
    sample's position on the sampling grid when samples are missing (default: 0, 1, 2, ...).
    """
    t = np.asarray(times_ms, dtype=float)
    index = np.arange(t.size, dtype=float) if sample_index is None else np.asarray(sample_index, dtype=float)
    segments = segments or [slice(0, t.size)]
    num = 0.0
    den = 0.0
    for seg in segments:
        ts, n = t[seg], index[seg]
        if ts.size < 2:
            continue
        n_c = n - n.mean()
        num += float(np.dot(n_c, ts - ts.mean()))
        den += float(np.dot(n_c, n_c))
    if den == 0.0:
        return float("nan")
    return num / den


def _lowpass_taps(cutoff_hz: float, fs: float, n_taps: int) -> np.ndarray:
    m = (n_taps - 1) / 2
    k = np.arange(n_taps) - m
    fc = cutoff_hz / fs
    return 2 * fc * np.sinc(2 * fc * k) * np.hamming(n_taps)


def qrs_bandpass_taps(fs: float) -> np.ndarray:
    """Linear-phase FIR bandpass (difference of two Hamming-windowed sinc lowpasses), odd length."""
    n_taps = int(np.ceil(3.3 * fs / FIR_TRANSITION_HZ)) | 1
    low, high = QRS_BAND_HZ
    return _lowpass_taps(high, fs, n_taps) - _lowpass_taps(low, fs, n_taps)


def _filter_zero_phase(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    # Symmetric odd-length taps + "valid" over a reflect-padded signal = zero group delay, no
    # circular wrap-around (which an FFT mask would introduce on a short record).
    half = taps.size // 2
    mode = "reflect" if x.size > half else "edge"
    padded = np.pad(x, half, mode=mode)
    return np.convolve(padded, taps, mode="valid")


def _derivative(x: np.ndarray, fs: float) -> np.ndarray:
    # Five-point central difference: y[n] = (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) * fs / 8.
    padded = np.pad(x, 2, mode="edge")
    kernel = np.array([1.0, 2.0, 0.0, -2.0, -1.0]) * fs / 8.0  # np.convolve flips the kernel
    return np.convolve(padded, kernel, mode="valid")


def _moving_window_integral(x: np.ndarray, width: int) -> np.ndarray:
    width = max(width, 1) | 1  # odd -> centred on the QRS
    return np.convolve(x, np.ones(width) / width, mode="same")


def _local_maxima(x: np.ndarray, min_distance: int) -> np.ndarray:
    """Local maxima, keeping the largest within any `min_distance` neighbourhood."""
    if x.size < 3:
        return np.array([], dtype=int)
    candidates = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    if candidates.size == 0:
        return candidates
    order = candidates[np.argsort(x[candidates])[::-1]]
    taken = np.zeros(x.size, dtype=bool)
    kept: list[int] = []
    for idx in order:
        lo, hi = max(idx - min_distance + 1, 0), min(idx + min_distance, x.size)
        if taken[lo:hi].any():
            continue
        taken[idx] = True
        kept.append(int(idx))
    return np.sort(np.asarray(kept, dtype=int))


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect R peaks in a uniformly sampled single-lead ECG.

    Args:
        signal: ECG amplitude samples (any unit, either polarity).
        fs: Sampling rate in Hz (>= MIN_SAMPLING_RATE_HZ).

    Returns:
        Fractional sample positions of the R peaks (sub-sample, via parabolic interpolation),
        strictly increasing. Empty if the input is too short or fs is too low.
    """
    x = np.asarray(signal, dtype=float)
    if fs < MIN_SAMPLING_RATE_HZ or x.size < int(fs * THRESHOLD_INIT_S):
        return np.array([], dtype=float)

    filtered = _filter_zero_phase(x - np.median(x), qrs_bandpass_taps(fs))
    slope = _derivative(filtered, fs)
    integrated = _moving_window_integral(slope**2, int(round(INTEGRATION_WINDOW_S * fs)))

    refractory = int(round(REFRACTORY_S * fs))
    t_wave_window = int(round(T_WAVE_WINDOW_S * fs))
    slope_half_window = int(round(INTEGRATION_WINDOW_S * fs / 2))
    candidates = _local_maxima(integrated, refractory)
    if candidates.size == 0:
        return np.array([], dtype=float)

    def max_slope(idx: int) -> float:
        lo, hi = max(idx - slope_half_window, 0), min(idx + slope_half_window + 1, slope.size)
        return float(np.max(np.abs(slope[lo:hi])))

    warmup = integrated[: int(fs * THRESHOLD_INIT_S)]
    signal_level = 0.25 * float(np.max(warmup))
    noise_level = 0.5 * float(np.mean(warmup))
    beats: list[int] = []

    def threshold() -> float:
        return noise_level + 0.25 * (signal_level - noise_level)

    def searchback(upto: int) -> None:
        nonlocal signal_level
        if len(beats) < 2:
            return
        recent_rr = np.diff(beats[-(SEARCHBACK_RR_HISTORY + 1) :])
        if upto - beats[-1] <= SEARCHBACK_RR_FACTOR * float(np.mean(recent_rr)):
            return
        lower = beats[-1] + refractory
        window = candidates[(candidates >= lower) & (candidates < upto)]
        window = window[integrated[window] > 0.5 * threshold()]
        if window.size == 0:
            return
        best = int(window[np.argmax(integrated[window])])
        beats.append(best)
        signal_level = 0.25 * float(integrated[best]) + 0.75 * signal_level

    for cand in candidates:
        idx = int(cand)
        searchback(idx)
        if beats and idx - beats[-1] < refractory:
            continue
        peak = float(integrated[idx])
        is_t_wave = (
            bool(beats)
            and idx - beats[-1] < t_wave_window
            and max_slope(idx) < T_WAVE_SLOPE_RATIO * max_slope(beats[-1])
        )
        if peak > threshold() and not is_t_wave:
            beats.append(idx)
            signal_level = 0.125 * peak + 0.875 * signal_level
        else:
            noise_level = 0.125 * peak + 0.875 * noise_level
    searchback(x.size)

    if not beats:
        return np.array([], dtype=float)
    return _refine_r_peaks(filtered, np.asarray(beats, dtype=int), fs)


def _refine_r_peaks(filtered: np.ndarray, coarse: np.ndarray, fs: float) -> np.ndarray:
    """Move each integrator peak to the signed band-passed extremum, then to sub-sample precision.

    Why sub-sample matters: rounding a peak to the sample grid adds error with sigma = T / sqrt(12)
    (~2.2 ms at 130 Hz). A successive RR difference RR[i+1] - RR[i] = t[i+2] - 2 t[i+1] + t[i]
    combines three peak times, so its error variance is (1 + 4 + 1) sigma^2 = 6 sigma^2 (~5.4 ms
    RMS at 130 Hz). That error is independent of the true variability, so it adds in quadrature
    and biases RMSSD upward: a true 20 ms reads ~sqrt(20^2 + 5.4^2) ~ 20.7 ms (+3.5%). Parabolic
    interpolation through the peak sample and its two neighbours shrinks sigma well below T/sqrt(12).
    """
    half = int(round(REFINE_HALF_WINDOW_S * fs))
    n = filtered.size

    # Lead orientation on a wrist device is unknown: pick the dominant QRS polarity once.
    extremes = []
    for idx in coarse:
        seg = filtered[max(idx - half, 0) : min(idx + half + 1, n)]
        extremes.append(seg[np.argmax(np.abs(seg))])
    polarity = 1.0 if np.median(extremes) >= 0 else -1.0
    oriented = polarity * filtered

    refined: list[float] = []
    for idx in coarse:
        lo, hi = max(idx - half, 0), min(idx + half + 1, n)
        k = lo + int(np.argmax(oriented[lo:hi]))
        offset = 0.0
        if 0 < k < n - 1:
            a, b, c = oriented[k - 1], oriented[k], oriented[k + 1]
            denom = a - 2 * b + c
            if denom < 0:
                offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        pos = k + offset
        if not refined or pos - refined[-1] >= REFRACTORY_S * fs:
            refined.append(pos)
    return np.asarray(refined, dtype=float)


def flag_rr_artifacts(rr_ms: np.ndarray) -> np.ndarray:
    """Return a mask that is True for RR intervals to keep (physiological and non-ectopic)."""
    rr = np.asarray(rr_ms, dtype=float)
    keep = (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
    if rr.size < 3:
        return keep
    # Reference = median of the in-bounds *neighbours* in a window of ECTOPIC_MEDIAN_WINDOW, shifted
    # (not truncated) at the record edges so every interval gets the same number of neighbours. An
    # interval must not vote for itself: with edge padding, an artifact at the start of a record
    # would be its own reference.
    width = min(ECTOPIC_MEDIAN_WINDOW, rr.size)
    in_bounds = keep.copy()
    for i in range(rr.size):
        start = min(max(i - width // 2, 0), rr.size - width)
        neighbours = np.r_[start:i, i + 1 : start + width]
        neighbours = neighbours[in_bounds[neighbours]]
        if neighbours.size == 0:
            continue
        reference = float(np.median(rr[neighbours]))
        keep[i] &= abs(rr[i] - reference) <= ECTOPIC_TOLERANCE * reference
    return keep


def successive_rr_differences_ms(rr_ms: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Successive differences RR[i+1] - RR[i] over pairs where both intervals are kept.

    Rejected intervals are dropped, never interpolated, and no difference spans them.
    """
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return np.array([], dtype=float)
    mask = np.ones(rr.size, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    pair_ok = mask[1:] & mask[:-1]
    return np.diff(rr)[pair_ok]


def calculate_rmssd_from_rr(
    rr_segments: Sequence[np.ndarray],
    keep_segments: Sequence[np.ndarray] | None = None,
) -> float:
    """RMSSD (ms) from true beat-to-beat RR intervals, differencing only within each segment.

    Returns NaN when no valid successive pair exists.
    """
    diffs = [
        successive_rr_differences_ms(rr, None if keep_segments is None else keep_segments[i])
        for i, rr in enumerate(rr_segments)
    ]
    pooled = np.concatenate(diffs) if diffs else np.array([], dtype=float)
    if pooled.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(pooled**2)))


@dataclass(frozen=True)
class RRInterval:
    """One beat-to-beat interval. `end_time_ms` is the closing beat, ms from recording start."""

    end_time_ms: float
    rr_ms: float
    segment: int
    is_artifact: bool
    is_low_quality: bool


@dataclass
class EcgAnalysis:
    sampling_rate_hz: float
    beat_times_ms: list[float] = field(default_factory=list)
    rr_intervals: list[RRInterval] = field(default_factory=list)
    rmssd_ms: float = float("nan")
    skipped_reason: str | None = None


def _split_segments(times_ms: np.ndarray, excluded: np.ndarray, max_gap_ms: float) -> list[slice]:
    """Contiguous runs of non-excluded samples with no time gap longer than `max_gap_ms`."""
    breaks = np.zeros(times_ms.size, dtype=bool)
    breaks[1:] = np.diff(times_ms) > max_gap_ms
    segments: list[slice] = []
    start: int | None = None
    for i in range(times_ms.size):
        if excluded[i] or breaks[i]:
            if start is not None:
                segments.append(slice(start, i))
            start = None if excluded[i] else i
        elif start is None:
            start = i
    if start is not None:
        segments.append(slice(start, times_ms.size))
    return segments


def analyze_ecg(
    times_ms: np.ndarray,
    amplitude: np.ndarray,
    no_contact: np.ndarray | None = None,
    low_quality: np.ndarray | None = None,
) -> EcgAnalysis:
    """Detect beats and derive RR intervals + RMSSD from a timestamped single-lead ECG.

    Args:
        times_ms: Sample times in ms from recording start (may be integer-rounded).
        amplitude: ECG samples. NaN marks a missing sample: holes up to MAX_BRIDGED_GAP_S are
            interpolated, longer ones split the recording.
        no_contact: Per-sample mask; True samples are excluded from detection and split segments.
        low_quality: Per-sample mask; beats there are detected and returned but their RR intervals
            are flagged and excluded from the RMSSD.
    """
    t = np.asarray(times_ms, dtype=float)
    x = np.asarray(amplitude, dtype=float)
    excluded = np.zeros(t.size, dtype=bool) if no_contact is None else np.asarray(no_contact, dtype=bool)
    low = np.zeros(t.size, dtype=bool) if low_quality is None else np.asarray(low_quality, dtype=bool)
    # Missing samples (NaN) are dropped here and bridged or split on below, like any other hole.
    present = np.isfinite(x) & np.isfinite(t)
    t, x, excluded, low = t[present], x[present], excluded[present], low[present]
    order = np.argsort(t, kind="stable")
    t, x, excluded, low = t[order], x[order], excluded[order], low[order]

    if t.size < 2:
        return EcgAnalysis(sampling_rate_hz=float("nan"), skipped_reason="too_few_samples")

    rough_period = float(np.median(np.diff(t)))
    if not rough_period > 0:
        return EcgAnalysis(sampling_rate_hz=float("nan"), skipped_reason="no_time_base")
    # A hole of k missing samples is a step of (k + 1) periods; half a period of slack for jitter.
    max_gap_ms = max(MAX_BRIDGED_GAP_S * 1000.0, rough_period) + 0.5 * rough_period
    segments = _split_segments(t, excluded, max_gap_ms)
    # Position of every sample on its segment's sampling grid (accounts for missing samples). Round
    # each step, not the elapsed time: the median step of integer-ms stamps can be off by up to
    # 0.5 ms (8 ms at 130 Hz), which would drift by hundreds of samples over a recording.
    grid_index = np.zeros(t.size)
    for seg in segments:
        grid_index[seg] = np.concatenate([[0.0], np.cumsum(np.rint(np.diff(t[seg]) / rough_period))])
    period_ms = estimate_sampling_period_ms(t, segments, grid_index)
    if not np.isfinite(period_ms) or period_ms <= 0:
        return EcgAnalysis(sampling_rate_hz=float("nan"), skipped_reason="no_time_base")
    fs = 1000.0 / period_ms
    if fs < MIN_SAMPLING_RATE_HZ:
        return EcgAnalysis(sampling_rate_hz=fs, skipped_reason="sampling_rate_too_low")

    analysis = EcgAnalysis(sampling_rate_hz=fs)
    rr_segments: list[np.ndarray] = []
    keep_segments: list[np.ndarray] = []
    for seg_id, seg in enumerate(segments):
        ts, n = t[seg], grid_index[seg]
        if ts.size < 2 or (n[-1] + 1) * period_ms < MIN_SEGMENT_S * 1000.0:
            continue
        # Uniform grid for the segment: bridged holes are linearly interpolated. Beat times come
        # from the fitted time base (intercept per segment, pooled slope), so they are not
        # quantised to the integer-ms timestamps.
        grid = np.arange(int(n[-1]) + 1, dtype=float)
        xs = np.interp(grid, n, x[seg])
        lows = low[seg][np.clip(np.searchsorted(n, grid), 0, n.size - 1)]
        intercept = float(ts.mean() - period_ms * n.mean())
        peaks = detect_r_peaks(xs, fs)
        if peaks.size == 0:
            continue
        beat_times = intercept + period_ms * peaks
        beat_low = lows[np.clip(np.rint(peaks).astype(int), 0, grid.size - 1)]
        analysis.beat_times_ms.extend(beat_times.tolist())

        rr = np.diff(beat_times)
        if rr.size == 0:
            continue
        artifact = ~flag_rr_artifacts(rr)
        rr_low = beat_low[1:] | beat_low[:-1]
        for end_time, value, is_art, is_low in zip(beat_times[1:], rr, artifact, rr_low, strict=True):
            analysis.rr_intervals.append(
                RRInterval(
                    end_time_ms=float(end_time),
                    rr_ms=float(value),
                    segment=seg_id,
                    is_artifact=bool(is_art),
                    is_low_quality=bool(is_low),
                )
            )
        rr_segments.append(rr)
        keep_segments.append(~artifact & ~rr_low)

    if not analysis.beat_times_ms:
        analysis.skipped_reason = "no_beats_detected"
        return analysis
    artifacts = [iv.is_artifact for iv in analysis.rr_intervals]
    artifact_fraction = float(np.mean(artifacts)) if artifacts else 0.0
    if artifact_fraction > MAX_ARTIFACT_FRACTION:
        analysis.skipped_reason = "too_many_artifacts"
        return analysis
    analysis.rmssd_ms = calculate_rmssd_from_rr(rr_segments, keep_segments)
    return analysis
