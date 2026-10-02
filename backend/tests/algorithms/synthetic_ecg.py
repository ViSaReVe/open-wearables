"""Synthetic single-lead ECG with known R-peak times (ground truth for detector tests).

Beats are sums of Gaussians (P, Q, R, S, T) placed at R times drawn from an RR series with
respiratory sinus arrhythmia + jitter, so the true RR intervals and RMSSD are known exactly.
Optional baseline wander, white noise, mains hum and band-limited EMG-like noise.
"""

from dataclasses import dataclass

import numpy as np

# (amplitude relative to R, offset from R in s, sigma in s). Q-S spans ~80 ms.
_WAVES: dict[str, tuple[float, float, float]] = {
    "P": (0.15, -0.20, 0.025),
    "Q": (-0.12, -0.025, 0.008),
    "R": (1.0, 0.0, 0.010),
    "S": (-0.25, 0.025, 0.008),
    "T": (0.30, 0.30, 0.050),
}


@dataclass(frozen=True)
class SyntheticEcg:
    signal_mv: np.ndarray
    fs: float
    r_times_s: np.ndarray  # true R-peak times (continuous, not on the sample grid)

    @property
    def times_ms(self) -> np.ndarray:
        """Sample timestamps as a provider would deliver them: integer ms from recording start."""
        return np.rint(np.arange(self.signal_mv.size) * 1000.0 / self.fs)

    @property
    def true_rr_ms(self) -> np.ndarray:
        return np.diff(self.r_times_s) * 1000.0

    @property
    def true_rmssd_ms(self) -> float:
        return float(np.sqrt(np.mean(np.diff(self.true_rr_ms) ** 2)))


def make_rr_series(
    duration_s: float,
    mean_rr_ms: float = 1000.0,
    rsa_amplitude_ms: float = 40.0,
    rsa_hz: float = 0.25,
    jitter_ms: float = 15.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """RR intervals (ms) covering `duration_s`, modulated by a sinusoidal RSA term + white jitter."""
    rng = rng or np.random.default_rng(0)
    rr: list[float] = []
    t = 0.0
    while t < duration_s * 1000.0:
        value = mean_rr_ms + rsa_amplitude_ms * np.sin(2 * np.pi * rsa_hz * t / 1000.0) + rng.normal(0, jitter_ms)
        rr.append(float(value))
        t += value
    return np.asarray(rr)


def synthesize_ecg(
    rr_ms: np.ndarray,
    fs: float = 130.0,
    r_amplitude_mv: float = 1.0,
    noise_mv: float = 0.02,
    baseline_mv: float = 0.15,
    baseline_hz: float = 0.3,
    mains_mv: float = 0.0,
    mains_hz: float = 50.0,
    emg_mv: float = 0.0,
    polarity: float = 1.0,
    t_wave_scale: float = 1.0,
    lead_in_s: float = 0.8,
    rng: np.random.Generator | None = None,
) -> SyntheticEcg:
    """Render an ECG waveform sampled at `fs` with R peaks at lead_in + cumsum(rr)."""
    rng = rng or np.random.default_rng(1)
    r_times = lead_in_s + np.concatenate([[0.0], np.cumsum(rr_ms) / 1000.0])
    duration = r_times[-1] + 0.8
    t = np.arange(int(duration * fs)) / fs
    r_times = r_times[r_times < t[-1] - 0.5]

    ecg = np.zeros_like(t)
    for r in r_times:
        for name, (amp, offset, sigma) in _WAVES.items():
            scale = t_wave_scale if name == "T" else 1.0
            # Render each wave over +/- 6 sigma only (exp(-18) ~ 1e-8 of the peak beyond that).
            lo = max(int((r + offset - 6 * sigma) * fs), 0)
            hi = min(int((r + offset + 6 * sigma) * fs) + 2, t.size)
            ecg[lo:hi] += scale * amp * np.exp(-0.5 * ((t[lo:hi] - (r + offset)) / sigma) ** 2)
    ecg *= polarity * r_amplitude_mv

    ecg += baseline_mv * np.sin(2 * np.pi * baseline_hz * t + rng.uniform(0, 2 * np.pi))
    ecg += rng.normal(0, noise_mv, t.size) if noise_mv > 0 else 0.0
    if mains_mv > 0:
        ecg += mains_mv * np.sin(2 * np.pi * mains_hz * t)
    if emg_mv > 0:
        # Band-limited (~20-60 Hz) muscle noise: white noise minus its smoothed version.
        white = rng.normal(0, 1, t.size)
        width = max(int(fs / 20), 1)
        hp = white - np.convolve(white, np.ones(width) / width, mode="same")
        ecg += emg_mv * hp / np.std(hp)
    return SyntheticEcg(signal_mv=ecg, fs=fs, r_times_s=r_times)


def match_peaks(
    true_s: np.ndarray,
    detected_s: np.ndarray,
    tolerance_s: float = 0.150,
) -> tuple[int, int, int, np.ndarray]:
    """Greedy one-to-one matching within +/- tolerance (AAMI EC57 convention: 150 ms).

    Returns (true_positives, false_positives, false_negatives, timing_errors_s of matches).
    """
    used = np.zeros(detected_s.size, dtype=bool)
    errors: list[float] = []
    for r in true_s:
        if detected_s.size == 0:
            break
        dist = np.abs(detected_s - r)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= tolerance_s:
            used[j] = True
            errors.append(float(detected_s[j] - r))
    tp = int(used.sum())
    return tp, int(detected_s.size - tp), int(true_s.size - tp), np.asarray(errors)
