"""Unit tests for app.algorithms.ecg (R-peak detection, RR derivation, RMSSD).

Ground truth comes from the synthetic generator in tests/algorithms/synthetic_ecg.py, which places
QRS complexes at known (continuous-time) R-peak instants. Pure numpy — no database, no factories.
"""

import numpy as np
import pytest

from app.algorithms.ecg import (
    MAX_ARTIFACT_FRACTION,
    analyze_ecg,
    calculate_rmssd_from_rr,
    detect_r_peaks,
    estimate_sampling_period_ms,
    flag_rr_artifacts,
    successive_rr_differences_ms,
)
from app.algorithms.resilience import calculate_rmssd
from tests.algorithms.synthetic_ecg import SyntheticEcg, make_rr_series, match_peaks, synthesize_ecg

# Noise levels used across the suite. "noisy" ~ R amplitude 1 mV with 0.1 mV white noise,
# 0.05 mV EMG-like and mains components and 0.3 mV baseline wander.
NOISE_LEVELS: dict[str, dict[str, float]] = {
    "clean": {"noise_mv": 0.02},
    "noisy": {"noise_mv": 0.10, "emg_mv": 0.05, "mains_mv": 0.05, "baseline_mv": 0.3},
}
SAMPLING_RATES = (130.0, 250.0, 500.0)
SEEDS = range(5)


def _recording(seed: int, fs: float, duration_s: float = 60.0, **kwargs: float) -> SyntheticEcg:
    rng = np.random.default_rng(seed)
    rr = make_rr_series(
        duration_s,
        mean_rr_ms=rng.uniform(600, 1200),
        rsa_amplitude_ms=rng.uniform(10, 50),
        jitter_ms=rng.uniform(5, 20),
        rng=rng,
    )
    return synthesize_ecg(rr, fs=fs, rng=rng, **kwargs)


def _detection_stats(fs: float, **kwargs: float) -> tuple[float, float, list[float]]:
    tp = fp = fn = 0
    rmssd_errors: list[float] = []
    for seed in SEEDS:
        ecg = _recording(seed, fs, **kwargs)
        analysis = analyze_ecg(ecg.times_ms, ecg.signal_mv)
        t, f, n, _ = match_peaks(ecg.r_times_s, np.asarray(analysis.beat_times_ms) / 1000.0)
        tp, fp, fn = tp + t, fp + f, fn + n
        rmssd_errors.append(analysis.rmssd_ms - ecg.true_rmssd_ms)
    return tp / (tp + fp), tp / (tp + fn), rmssd_errors


# ---------------------------------------------------------------------------
# Time base
# ---------------------------------------------------------------------------


class TestEstimateSamplingPeriod:
    @pytest.mark.parametrize("fs", SAMPLING_RATES)
    def test_recovers_period_from_integer_ms_timestamps(self, fs: float) -> None:
        """Timestamps rounded to whole ms (7/8 ms steps at 130 Hz) still give the exact period."""
        times = np.rint(np.arange(4000) * 1000.0 / fs)
        assert estimate_sampling_period_ms(times) == pytest.approx(1000.0 / fs, abs=1e-3)

    def test_pooled_slope_ignores_gap_between_segments(self) -> None:
        times = np.concatenate([np.arange(1000) * 4.0, 10_000 + np.arange(1000) * 4.0])
        segments = [slice(0, 1000), slice(1000, 2000)]
        assert estimate_sampling_period_ms(times, segments) == pytest.approx(4.0)

    def test_single_sample_is_nan(self) -> None:
        assert np.isnan(estimate_sampling_period_ms(np.array([0.0])))


# ---------------------------------------------------------------------------
# R-peak detection
# ---------------------------------------------------------------------------


class TestDetectRPeaks:
    @pytest.mark.parametrize("fs", SAMPLING_RATES)
    @pytest.mark.parametrize("noise", NOISE_LEVELS)
    def test_precision_and_recall(self, fs: float, noise: str) -> None:
        precision, recall, _ = _detection_stats(fs, **NOISE_LEVELS[noise])
        assert precision >= 0.98
        assert recall >= 0.99

    @pytest.mark.parametrize("fs", SAMPLING_RATES)
    def test_inverted_lead(self, fs: float) -> None:
        """Wrist lead orientation is unknown; a negative QRS must be found at the same instants."""
        precision, recall, _ = _detection_stats(fs, polarity=-1.0, **NOISE_LEVELS["noisy"])
        assert precision >= 0.98
        assert recall >= 0.99

    def test_tall_t_wave_not_counted_as_beat(self) -> None:
        precision, recall, _ = _detection_stats(250.0, t_wave_scale=2.5, **NOISE_LEVELS["clean"])
        assert precision == 1.0
        assert recall == 1.0

    def test_sub_sample_timing_beats_the_sample_grid(self) -> None:
        """Parabolic refinement: timing jitter far below T/sqrt(12) (2.2 ms at 130 Hz)."""
        ecg = _recording(0, 130.0, noise_mv=0.0, baseline_mv=0.0)
        peaks_s = detect_r_peaks(ecg.signal_mv, ecg.fs) / ecg.fs
        _, _, _, errors = match_peaks(ecg.r_times_s, peaks_s)
        grid_sigma_s = (1 / ecg.fs) / np.sqrt(12)
        assert np.std(errors) < 0.1 * grid_sigma_s

    def test_grid_quantisation_biases_rmssd_upward_by_six_sigma_squared(self) -> None:
        """Rounding R peaks to samples adds 6 sigma^2 = T^2 / 2 to RMSSD^2 (in quadrature).

        The measured excess is mean(d_noise^2) + 2 mean(d_true * d_noise); the cross term has
        std ~ 2 sqrt(RMSSD^2 * 6 sigma^2 / N), so a low true RMSSD (jitter only, ~4 ms) and a long
        record keep it small relative to T^2 / 2.
        """
        fs = 130.0
        rr = make_rr_series(1800.0, mean_rr_ms=900.0, rsa_amplitude_ms=0.0, jitter_ms=3.0)
        ecg = synthesize_ecg(rr, fs=fs, noise_mv=0.0, baseline_mv=0.0)
        positions = detect_r_peaks(ecg.signal_mv, fs)
        rounded_rmssd = calculate_rmssd_from_rr([np.diff(np.rint(positions)) * 1000.0 / fs])
        refined_rmssd = calculate_rmssd_from_rr([np.diff(positions) * 1000.0 / fs])
        true_sq = ecg.true_rmssd_ms**2
        predicted_excess = (1000.0 / fs) ** 2 / 2
        assert rounded_rmssd**2 - true_sq == pytest.approx(predicted_excess, rel=0.15)
        assert abs(refined_rmssd**2 - true_sq) < 0.1 * predicted_excess

    def test_sampling_rate_below_minimum_returns_empty(self) -> None:
        assert detect_r_peaks(np.zeros(1000), fs=40.0).size == 0

    def test_too_short_returns_empty(self) -> None:
        assert detect_r_peaks(np.zeros(100), fs=130.0).size == 0

    def test_flat_line_has_no_beats(self) -> None:
        assert detect_r_peaks(np.zeros(1300), fs=130.0).size == 0


# ---------------------------------------------------------------------------
# RR cleaning and RMSSD
# ---------------------------------------------------------------------------


class TestFlagRrArtifacts:
    def test_out_of_physiological_bounds(self) -> None:
        keep = flag_rr_artifacts(np.array([200.0, 1000.0, 2500.0]))
        assert keep.tolist() == [False, True, False]

    def test_ectopic_beat_and_compensatory_pause(self) -> None:
        rr = np.array([1000.0, 1000.0, 600.0, 1400.0, 1000.0, 1000.0])
        assert flag_rr_artifacts(rr).tolist() == [True, True, False, False, True, True]

    def test_artifact_at_record_start_is_caught(self) -> None:
        """Regression: an edge-padded median let the first interval be its own reference."""
        rr = np.array([587.0, 1177.0, 1163.0, 1150.0, 1160.0])
        assert flag_rr_artifacts(rr).tolist() == [False, True, True, True, True]

    def test_respiratory_variation_is_kept(self) -> None:
        rr = 1000.0 + 50.0 * np.sin(np.arange(40) * 2 * np.pi / 4)
        assert flag_rr_artifacts(rr).all()


class TestRmssdFromRr:
    def test_known_value(self) -> None:
        # diffs: 20, -20, 20 -> RMSSD 20
        assert calculate_rmssd_from_rr([np.array([1000.0, 1020.0, 1000.0, 1020.0])]) == pytest.approx(20.0)

    def test_no_difference_across_rejected_interval(self) -> None:
        rr = np.array([1000.0, 1010.0, 400.0, 1000.0, 1010.0])
        keep = np.array([True, True, False, True, True])
        assert successive_rr_differences_ms(rr, keep).tolist() == [10.0, 10.0]

    def test_no_difference_across_segments(self) -> None:
        """Two flat segments at different rates: differencing across the gap would add 200 ms."""
        segments = [np.array([1000.0, 1000.0, 1000.0]), np.array([800.0, 800.0, 800.0])]
        assert calculate_rmssd_from_rr(segments) == 0.0

    def test_fewer_than_two_intervals_is_nan(self) -> None:
        assert np.isnan(calculate_rmssd_from_rr([np.array([1000.0])]))
        assert np.isnan(calculate_rmssd_from_rr([]))


# ---------------------------------------------------------------------------
# Full recording analysis
# ---------------------------------------------------------------------------


class TestAnalyzeEcg:
    @pytest.mark.parametrize("fs", SAMPLING_RATES)
    @pytest.mark.parametrize(("noise", "tolerance_ms"), [("clean", 0.5), ("noisy", 1.0)])
    def test_rmssd_matches_ground_truth(self, fs: float, noise: str, tolerance_ms: float) -> None:
        _, _, errors = _detection_stats(fs, **NOISE_LEVELS[noise])
        assert np.nanmax(np.abs(errors)) < tolerance_ms

    def test_estimates_sampling_rate(self) -> None:
        ecg = _recording(0, 130.0)
        assert analyze_ecg(ecg.times_ms, ecg.signal_mv).sampling_rate_hz == pytest.approx(130.0, rel=1e-4)

    def test_beat_times_are_ms_from_recording_start(self) -> None:
        ecg = _recording(0, 250.0, **NOISE_LEVELS["clean"])
        analysis = analyze_ecg(ecg.times_ms, ecg.signal_mv)
        rr_ends = [iv.end_time_ms for iv in analysis.rr_intervals]
        assert rr_ends == analysis.beat_times_ms[1:]
        assert analysis.beat_times_ms[0] == pytest.approx(ecg.r_times_s[0] * 1000.0, abs=5.0)

    def test_no_contact_splits_segments_and_no_rr_spans_the_gap(self) -> None:
        ecg = _recording(2, 130.0, duration_s=40.0, **NOISE_LEVELS["clean"])
        times = ecg.times_ms
        no_contact = (times >= 15_000) & (times < 22_000)
        analysis = analyze_ecg(times, ecg.signal_mv, no_contact=no_contact)

        assert {iv.segment for iv in analysis.rr_intervals} == {0, 1}
        assert not any(15_000 <= t < 22_000 for t in analysis.beat_times_ms)
        assert max(iv.rr_ms for iv in analysis.rr_intervals) < 2000.0
        assert analysis.rmssd_ms == pytest.approx(ecg.true_rmssd_ms, abs=5.0)

    def test_short_segment_is_skipped(self) -> None:
        ecg = _recording(0, 130.0, duration_s=12.0, **NOISE_LEVELS["clean"])
        no_contact = ecg.times_ms >= 4_000
        analysis = analyze_ecg(ecg.times_ms, ecg.signal_mv, no_contact=no_contact)
        assert analysis.beat_times_ms == []
        assert analysis.skipped_reason == "no_beats_detected"

    def test_low_quality_beats_kept_but_excluded_from_rmssd(self) -> None:
        ecg = _recording(3, 250.0, **NOISE_LEVELS["clean"])
        times = ecg.times_ms
        low = times >= 30_000
        full = analyze_ecg(times, ecg.signal_mv)
        gated = analyze_ecg(times, ecg.signal_mv, low_quality=low)

        assert gated.beat_times_ms == full.beat_times_ms
        flagged = [iv for iv in gated.rr_intervals if iv.is_low_quality]
        assert flagged
        assert all(iv.end_time_ms >= 30_000 for iv in flagged)
        first_half = np.asarray([iv.rr_ms for iv in gated.rr_intervals if not iv.is_low_quality])
        assert gated.rmssd_ms == pytest.approx(calculate_rmssd_from_rr([first_half]))

    def test_nan_samples_are_gaps(self) -> None:
        ecg = _recording(0, 130.0, duration_s=40.0, **NOISE_LEVELS["clean"])
        signal = ecg.signal_mv.copy()
        signal[(ecg.times_ms >= 20_000) & (ecg.times_ms < 21_000)] = np.nan
        analysis = analyze_ecg(ecg.times_ms, signal)
        assert len({iv.segment for iv in analysis.rr_intervals}) == 2

    def test_low_sampling_rate_is_skipped(self) -> None:
        times = np.arange(0, 30_000, 40.0)  # 25 Hz
        analysis = analyze_ecg(times, np.zeros(times.size))
        assert analysis.skipped_reason == "sampling_rate_too_low"
        assert analysis.sampling_rate_hz == pytest.approx(25.0)

    def test_too_many_artifacts_withholds_rmssd(self) -> None:
        """Pure noise: whatever is 'detected' is irregular, so no RMSSD is reported."""
        rng = np.random.default_rng(0)
        times = np.rint(np.arange(130 * 60) * 1000.0 / 130.0)
        analysis = analyze_ecg(times, rng.normal(0, 1, times.size))
        artifact_fraction = np.mean([iv.is_artifact for iv in analysis.rr_intervals])
        assert artifact_fraction > MAX_ARTIFACT_FRACTION
        assert analysis.skipped_reason == "too_many_artifacts"
        assert np.isnan(analysis.rmssd_ms)

    def test_empty_input(self) -> None:
        analysis = analyze_ecg(np.array([]), np.array([]))
        assert analysis.skipped_reason == "too_few_samples"


# ---------------------------------------------------------------------------
# Negative control: the 60000/HR path this replaces
# ---------------------------------------------------------------------------


class TestNegativeControlHrDerivedRmssd:
    """RMSSD via `60000/HR` on windowed bpm (app.algorithms.resilience.calculate_rmssd) vs truth.

    Windowed HR is a boxcar low-pass of the beat train: it removes respiratory sinus arrhythmia
    (0.15-0.4 Hz), which is what RMSSD measures, and integer-bpm rounding adds a noise floor.
    """

    @staticmethod
    def _windowed_bpm(r_times_s: np.ndarray, window_s: float) -> list[float]:
        rr_ms = np.diff(r_times_s) * 1000.0
        ends = r_times_s[1:]
        bpm: list[float] = []
        for start in np.arange(ends[0], ends[-1] - window_s, window_s):
            in_window = rr_ms[(ends >= start) & (ends < start + window_s)]
            if in_window.size:
                bpm.append(float(np.round(60000.0 / in_window.mean())))
        return bpm

    @pytest.mark.parametrize("window_s", [5.0, 15.0, 60.0])
    def test_hr_path_underestimates_while_beat_path_matches(self, window_s: float) -> None:
        ecg = _recording(4, 250.0, duration_s=600.0, **NOISE_LEVELS["noisy"])
        true_rmssd = ecg.true_rmssd_ms

        hr_rmssd = calculate_rmssd(self._windowed_bpm(ecg.r_times_s, window_s))
        beat_rmssd = analyze_ecg(ecg.times_ms, ecg.signal_mv).rmssd_ms

        assert abs(beat_rmssd - true_rmssd) / true_rmssd < 0.05
        assert abs(hr_rmssd - true_rmssd) / true_rmssd > 0.5
