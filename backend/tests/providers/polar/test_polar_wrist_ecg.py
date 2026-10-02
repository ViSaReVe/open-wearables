"""Tests for Polar wrist ECG normalization (GET /v3/users/wrist-ecg).

Payloads:
- ``DOCUMENTED_EXAMPLE`` is the response example from the Polar AccessLink API reference, verbatim.
  Its values are placeholders (one sample, ``rri_ms`` = 100), so it only validates the shape.
- ``_synthetic_payload`` renders a synthetic ECG (tests/algorithms/synthetic_ecg.py) into the same
  documented shape. It is NOT a real device recording: the real sampling rate, the meaning of
  ``rri_ms`` and the quality-measurement semantics are undocumented.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from app.schemas.enums import SeriesType
from app.schemas.model_crud.activities import TimeSeriesSampleCreate
from app.services.providers.polar.data_247 import Polar247Data
from app.services.providers.polar.strategy import PolarStrategy
from tests.algorithms.synthetic_ecg import SyntheticEcg, make_rr_series, synthesize_ecg

DOCUMENTED_EXAMPLE: dict = {
    "source_device_id": "1111AAAA",
    "test_time": 1697787256,
    "time_zone_offset": 180,
    "average_heart_rate_bpm": 60,
    "heart_rate_variability_ms": 0,
    "heart_rate_variability_level": "ECG_HRV_LEVEL_NO_BASELINE",
    "rri_ms": 100,
    "pulse_transit_time_systolic_ms": 100,
    "pulse_transit_time_diastolic_ms": 100,
    "pulse_transit_time_quality_index": 100,
    "samples": [{"recording_time_delta_ms": 123, "amplitude_mv": 0.1}],
    "quality_measurements": [{"recording_time_delta_ms": 123, "quality_level": "ECG_QUALITY_HIGH"}],
}

TEST_TIME = 1_700_000_000


@pytest.fixture
def data_247() -> Polar247Data:
    data = PolarStrategy().data_247
    assert isinstance(data, Polar247Data)
    return data


@pytest.fixture
def ecg() -> SyntheticEcg:
    rr = make_rr_series(30.0, mean_rr_ms=900.0, rsa_amplitude_ms=40.0, jitter_ms=10.0, rng=np.random.default_rng(7))
    return synthesize_ecg(rr, fs=130.0, noise_mv=0.03, rng=np.random.default_rng(8))


def _synthetic_payload(ecg: SyntheticEcg, quality: list[tuple[int, str]] | None = None) -> dict:
    return {
        "source_device_id": "SYNTHETIC",
        "test_time": TEST_TIME,
        "time_zone_offset": 60,
        "average_heart_rate_bpm": 67,
        "heart_rate_variability_ms": 41.0,
        "heart_rate_variability_level": "ECG_HRV_LEVEL_USUAL",
        "rri_ms": 900.0,
        "samples": [
            {"recording_time_delta_ms": int(t), "amplitude_mv": round(float(a), 4)}
            for t, a in zip(ecg.times_ms, ecg.signal_mv, strict=True)
        ],
        "quality_measurements": [
            {"recording_time_delta_ms": t, "quality_level": level}
            for t, level in (quality if quality is not None else [(0, "ECG_QUALITY_HIGH")])
        ],
    }


def _of_type(samples: list[TimeSeriesSampleCreate], series_type: SeriesType) -> list[TimeSeriesSampleCreate]:
    return sorted((s for s in samples if s.series_type == series_type), key=lambda s: s.recorded_at)


def _offset_ms(sample: TimeSeriesSampleCreate) -> float:
    return (sample.recorded_at - datetime.fromtimestamp(TEST_TIME, tz=timezone.utc)).total_seconds() * 1000.0


class TestPolarWristEcgDocumentedExample:
    def test_vendor_values_unchanged(self, data_247: Polar247Data) -> None:
        samples = data_247.normalize_wrist_ecg([DOCUMENTED_EXAMPLE], uuid4())
        recorded_at = datetime.fromtimestamp(1697787256, tz=timezone.utc)

        (hrv,) = _of_type(samples, SeriesType.heart_rate_variability_rmssd)
        (hr,) = _of_type(samples, SeriesType.heart_rate)
        assert (hrv.value, hrv.recorded_at) == (0, recorded_at)
        assert (hr.value, hr.recorded_at) == (60, recorded_at)

    def test_quality_row_at_its_offset(self, data_247: Polar247Data) -> None:
        samples = data_247.normalize_wrist_ecg([DOCUMENTED_EXAMPLE], uuid4())
        (quality,) = _of_type(samples, SeriesType.ecg_signal_quality)
        assert quality.value == 3
        assert quality.recorded_at == datetime.fromtimestamp(1697787256, tz=timezone.utc) + timedelta(milliseconds=123)

    def test_single_sample_yields_no_rr(self, data_247: Polar247Data) -> None:
        samples = data_247.normalize_wrist_ecg([DOCUMENTED_EXAMPLE], uuid4())
        assert _of_type(samples, SeriesType.rr_interval) == []


class TestPolarWristEcgBeatToBeat:
    def test_vendor_values_kept_alongside_rr(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        samples = data_247.normalize_wrist_ecg([_synthetic_payload(ecg)], uuid4())
        assert [s.value for s in _of_type(samples, SeriesType.heart_rate_variability_rmssd)] == [41.0]
        assert [s.value for s in _of_type(samples, SeriesType.heart_rate)] == [67]
        assert _of_type(samples, SeriesType.rr_interval)

    def test_rr_rows_match_true_beats(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        rr_rows = _of_type(data_247.normalize_wrist_ecg([_synthetic_payload(ecg)], uuid4()), SeriesType.rr_interval)

        assert len(rr_rows) == ecg.true_rr_ms.size
        np.testing.assert_allclose([float(s.value) for s in rr_rows], ecg.true_rr_ms, atol=3.0)
        # Timestamped at the closing beat: test_time + R-peak time (constant morphology offset only).
        np.testing.assert_allclose([_offset_ms(s) for s in rr_rows], ecg.r_times_s[1:] * 1000.0, atol=5.0)

    def test_rmssd_from_persisted_rr_matches_truth(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        rr_rows = _of_type(data_247.normalize_wrist_ecg([_synthetic_payload(ecg)], uuid4()), SeriesType.rr_interval)
        rr = np.array([float(s.value) for s in rr_rows])
        assert float(np.sqrt(np.mean(np.diff(rr) ** 2))) == pytest.approx(ecg.true_rmssd_ms, abs=1.0)

    def test_rows_are_self_describing_for_contiguity(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        """Consecutive rows are adjacent beats iff their time gap equals the later row's value."""
        rr_rows = _of_type(data_247.normalize_wrist_ecg([_synthetic_payload(ecg)], uuid4()), SeriesType.rr_interval)
        gaps = np.diff([_offset_ms(s) for s in rr_rows])
        np.testing.assert_allclose(gaps, [float(s.value) for s in rr_rows[1:]], atol=0.01)

    def test_no_contact_stretch_has_no_beats(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        quality = [(0, "ECG_QUALITY_HIGH"), (12_000, "ECG_QUALITY_NO_CONTACT"), (18_000, "ECG_QUALITY_HIGH")]
        samples = data_247.normalize_wrist_ecg([_synthetic_payload(ecg, quality)], uuid4())

        rr_rows = _of_type(samples, SeriesType.rr_interval)
        offsets = [_offset_ms(s) for s in rr_rows]
        assert not any(12_000 <= t < 18_000 for t in offsets)
        assert any(t < 12_000 for t in offsets)
        assert any(t >= 18_000 for t in offsets)
        assert max(float(s.value) for s in rr_rows) < 2000.0
        assert [s.value for s in _of_type(samples, SeriesType.ecg_signal_quality)] == [3, 1, 3]

    def test_low_quality_beats_are_persisted(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        quality = [(0, "ECG_QUALITY_HIGH"), (15_000, "ECG_QUALITY_LOW")]
        rr_rows = _of_type(
            data_247.normalize_wrist_ecg([_synthetic_payload(ecg, quality)], uuid4()), SeriesType.rr_interval
        )
        assert any(_offset_ms(s) >= 15_000 for s in rr_rows)

    def test_duplicate_quality_offset_keeps_last(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        quality = [(0, "ECG_QUALITY_LOW"), (0, "ECG_QUALITY_HIGH")]
        samples = data_247.normalize_wrist_ecg([_synthetic_payload(ecg, quality)], uuid4())
        assert [s.value for s in _of_type(samples, SeriesType.ecg_signal_quality)] == [3]

    def test_samples_without_amplitude_are_ignored(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        payload = _synthetic_payload(ecg)
        for sample in payload["samples"][::50]:
            sample["amplitude_mv"] = None
        assert _of_type(data_247.normalize_wrist_ecg([payload], uuid4()), SeriesType.rr_interval)

    def test_analysis_failure_keeps_vendor_values(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        with (
            patch("app.services.providers.polar.data_247.analyze_ecg", side_effect=RuntimeError("boom")),
            patch("app.services.providers.polar.data_247.log_and_capture_error") as capture,
        ):
            samples = data_247.normalize_wrist_ecg([_synthetic_payload(ecg)], uuid4())

        capture.assert_called_once()
        assert {s.series_type for s in samples} == {
            SeriesType.heart_rate_variability_rmssd,
            SeriesType.heart_rate,
            SeriesType.ecg_signal_quality,
        }

    def test_missing_test_time_skipped(self, data_247: Polar247Data, ecg: SyntheticEcg) -> None:
        payload = _synthetic_payload(ecg)
        payload["test_time"] = None
        assert data_247.normalize_wrist_ecg([payload], uuid4()) == []
