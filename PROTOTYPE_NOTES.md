# PROTOTYPE NOTES — Polar wrist-ECG raw-signal preservation

Branch `polar-raw-ecg-prototype` (local; see "Status" at the end). Base: `main` @ `02563075`.
This is ECG-fidelity / raw-signal work. Wrist ECG is a spot check at a single `test_time`, so
nothing here feeds the overnight Resilience Score (FINDINGS.md 2.2 #4).

## 1. What was built

| Commit | Content |
|---|---|
| `21542a16` | `backend/app/algorithms/ecg.py` — numpy-only R-peak detector, RR derivation, artifact filter, RMSSD. `backend/tests/algorithms/synthetic_ecg.py` — synthetic ECG generator with known R-peak times. `backend/tests/algorithms/test_ecg_algorithms.py` |
| `4806d041` | Two new series types, the Polar normalize extension, `backend/tests/providers/polar/test_polar_wrist_ecg.py`, regenerated docs |

**Detector** (`ecg.py`). Pan-Tompkins pipeline: 5–15 Hz linear-phase FIR bandpass (difference of
Hamming-windowed sincs, ~1.1·fs taps, zero-phase via reflect-pad + `valid` convolution) → 5-point
central derivative → square → 150 ms centred moving-window integral → adaptive signal/noise levels
(0.125 / 0.875 updates, threshold at 25% between them) with 250 ms refractory, searchback at
1.66 × mean of the last 8 RR, and T-wave rejection (< 360 ms and < ½ of the previous QRS slope).
Then each beat is moved to the band-passed extremum (one polarity chosen per recording) and given a
sub-sample position by parabolic interpolation. All parameters are in seconds/Hz; the sampling rate
is a least-squares fit of the sample timestamps against grid index.

**RR / RMSSD.** RR intervals are kept only if inside 250–2000 ms (240–30 bpm) and within 20% of
the median of 4 in-bounds neighbours. Successive differences use only pairs of kept intervals in the
same segment, which avoids the cross-gap differencing noted in FINDINGS.md 1.4. RMSSD is withheld
(`too_many_artifacts`) when more than 20% of intervals are rejected.

**Persistence** (`polar/data_247.py` `normalize_wrist_ecg`). Polar's `heart_rate_variability_ms`
→ `heart_rate_variability_rmssd` and `average_heart_rate_bpm` → `heart_rate` are unchanged. Added
next to them:
- `rr_interval` (id 8, ms): one row per non-artifact interval at `test_time + closing-beat time`.
- `ecg_signal_quality` (id 9, score): one row per Polar quality change, 0 unknown / 1 no contact / 2 low / 3 high.

## 2. Measured detector performance (synthetic data only)

Grid: 10 seeds × 60 s per cell. Mean RR U(600, 1200) ms, 0.25 Hz RSA amplitude U(10, 50) ms, white
RR jitter U(5, 20) ms. R amplitude 1 mV. Peak matching ±150 ms (AAMI EC57 convention).
"RMSSD err" = estimated − true, over recordings where RMSSD was reported. Script in Appendix A.

Noise levels:
- **clean**: 0.02 mV white noise, 0.15 mV baseline wander at 0.3 Hz.
- **noisy**: 0.10 mV white noise, 0.05 mV EMG-like (~20–60 Hz), 0.05 mV 50 Hz mains, 0.3 mV wander.
- **vnoisy** (stress, not a pass bar): 0.20 mV white noise, 0.10 mV EMG-like, 0.10 mV mains, 0.5 mV wander.

| fs | noise | variant | Precision | Recall | RMSSD err mean / max abs (ms) | RMSSD withheld |
|---|---|---|---|---|---|---|
| 130 | clean | normal / inverted / tall T | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | ≤ +0.01 / ≤ 0.24 | 0/10 each |
| 130 | noisy | normal | 0.9985 | 1.000 | +0.20 / 0.68 | 0/10 |
| 130 | noisy | inverted | 0.9927 | 1.000 | +0.11 / 0.43 | 0/10 |
| 130 | noisy | tall T | 0.9840 | 0.9985 | +0.18 / 0.66 | 1/10 |
| 250 | clean | all three | 1.000 | 1.000 | ≤ 0.00 / ≤ 0.12 | 0/10 |
| 250 | noisy | all three | 1.000 | 1.000 | ≤ +0.14 / ≤ 0.55 | 0/10 |
| 500 | clean | all three | 1.000 | 1.000 | ≤ +0.02 / ≤ 0.15 | 0/10 |
| 500 | noisy | all three | 1.000 | 1.000 | ≤ +0.12 / ≤ 0.43 | 0/10 |
| 130 | vnoisy | normal / inverted / tall T | 0.70 / 0.72 / 0.68 | 0.94 / 0.93 / 0.94 | +1.11 / 3.69 on the 3 reported | 7/10, 10/10, 9/10 |
| 250 | vnoisy | normal / inverted / tall T | 0.975 / 0.888 / 0.960 | 0.994 / 0.975 / 0.994 | up to **23.6** (inverted), 21.4 (tall T) | 0/10, 2/10, 0/10 |
| 500 | vnoisy | normal / inverted / tall T | 0.988 / 0.996 / 0.993 | 0.996 / 0.997 / 0.997 | ≤ 9.9 | 0/10 |

Other measurements:
- **Sampling grid quantisation (6σ²).** 1800 s at 130 Hz, noise-free, true RMSSD 4.26 ms.
  - Rounding R peaks to whole samples adds 27.3 ms² to RMSSD²; the prediction is T²/2 = 29.6 ms² (RMSSD 4.26 → 6.74 ms).
  - After parabolic refinement the excess is −0.03 ms² (RMSSD 4.258 ms).
  - Noise-free timing-error SD at 130 Hz is 0.03 ms, against T/√12 = 2.2 ms without interpolation.
- **Timing offset.** Every detected beat sits about −1.7 ms from the true R time. This is constant, so it cancels in RR intervals.
  - Cause is the synthetic QRS shape: the S wave is deeper than the Q wave, which pulls the band-passed peak earlier.
  - Evidence: with a symmetric Q/S the offset drops to −0.24 ms. Beat timestamps carry the offset, RR values do not.
- **Negative control** (`TestNegativeControlHrDerivedRmssd`): 600 s, 250 Hz, noisy, true RMSSD 45.31 ms.
  - Beat-to-beat path: 45.42 ms.
  - `calculate_rmssd` (`60000/HR` on windowed, integer bpm): 17.49 / 16.00 / 11.31 ms for 5 / 15 / 60 s windows.
- **Cost.** 4.2 ms per 60 s recording at 130 Hz and 6.9 ms at 500 Hz, single core, in the sync path.

**What these numbers do NOT show.** The synthetic QRS is a sum of Gaussians with fixed shape. There
are no real ectopic beats, no motion artifact, no electrode-contact transients, and no real wrist-lead
morphology (low amplitude, wide P/T). Real-data precision/recall is **unknown**.

## 3. Design decisions and tradeoffs

1. **numpy FIR, not an FFT mask.** It has linear phase and no wrap-around on short records. Cost is O(N·fs), which is fine at a few ms per recording. scipy was not needed anywhere.
2. **Sampling rate estimated, not configured.** Polar does not document it. The fit works from integer-ms timestamps (7/8 ms steps at 130 Hz); test: recovers fs to 1e-4 relative.
   - Below 50 Hz the recording is skipped, because the 15 Hz band edge needs margin under Nyquist.
3. **Grid index is rounded per step.** I first rounded elapsed time using the median step. That is wrong: at 130 Hz the median step is 8 ms against a true 7.69 ms, a ~150-sample drift over 30 s. The step-wise version is in the code.
4. **Short holes bridged.** Gaps of 25 ms or less (about the R-wave half-width) are linearly interpolated; longer gaps split the recording.
   - Found because a test with every 50th sample missing fragmented the record into 0.4 s pieces and produced zero beats.
   - Tradeoff: an interpolated hole on an R peak blunts it. The 25 ms limit is a heuristic.
5. **Quality policy (agreed before the build).**
   - NO_CONTACT: excluded from detection and splits the recording.
   - LOW: beats are detected and persisted, with the quality series next to them. Excluded from the RMSSD that `analyze_ecg` reports. Reversible.
   - Quality is assumed to hold until the next quality measurement (**unverified**).
6. **5 s minimum segment.** That is 2 s of threshold warm-up plus ≥ 3 beats. At very low heart rates (< ~40 bpm) a 5 s segment can hold fewer than 3 beats and gives no RMSSD pairs. Noted in the code comment.
7. **Artifact intervals are not persisted.**
   - Each row is stamped at its closing beat, so consecutive rows are adjacent beats exactly when their time gap equals the later row's value. Consumers can rebuild contiguous runs without a segment ID; there's a test for this.
   - Tradeoff: the reason a beat was rejected is lost from the DB. It is only visible in the analysis object and the logs.
8. **Derived RMSSD is not persisted.** `DataPointSeries` is unique on (data source, series type, `recorded_at`) (`models/data_point_series.py:22-29`). Our RMSSD at `test_time` would collide with Polar's. Options for maintainers: a separate series type (e.g. `heart_rate_variability_rmssd_beat`) or compute from `rr_interval` when read.
9. **Source attribution.** RR rows use `provider=source=polar`. The signal is Polar's, the detection is Open Wearables'. `TimeSeriesSampleCreate` has no "derived" marker; the only hint is the series description. **Question for maintainers.**
10. **Waveform analysis cannot drop vendor data.** Exceptions in the new path go to `log_and_capture_error` (Sentry); Polar's HR and HRV rows still save. Tested.
11. **Validation logging.** Each recording emits `action=wrist_ecg_beat_analysis` with our `sampling_rate_hz`, `mean_rr_ms` and `rmssd_ms` next to `polar_rri_ms` and `polar_hrv_ms`. One real payload settles items 1–3 in section 4.
12. **Not done, on purpose.**
    - (c) Raw waveform persistence: needs a new model and migration, and row volume depends on the unknown sampling rate. Follow-up.
    - Webhook events for the new types. Unmapped types are skipped silently (`constants/webhooks/events.py` docstring), so this changes nothing today.
    - MCP tool docstring: lists "common values" only.
13. **No migration.** Series type definitions are seeded at startup (`backend/scripts/start/app.sh:35` → `scripts/init/seed_series_types.py`). Precedent: #1242 added `active_time` without a migration. **Not runtime-checked here (no Docker).**
14. **Aggregation.** Both new types use AVG. Daily mean RR is meaningful; a daily mean of the quality code is only a rough summary.

## 4. What remains unknown

The AccessLink reference (polar.com/accesslink-api, `ecg-test-result`, `ecg-sample`,
`quality-measurement`) lists every field with Description "none". Its example payload is filler
(`rri_ms: 100` would be 600 bpm, `heart_rate_variability_ms: 0`, one sample). It is used verbatim as
a test that only checks the shape.

1. **Waveform sampling rate.** Not documented; estimated at runtime and logged.
2. **`rri_ms` semantics.** A single float; probably mean RR, not verified. Compare the logged `mean_rr_ms` with `polar_rri_ms`.
3. **Is `heart_rate_variability_ms` RMSSD?** The repo assumes yes (`data_247.py`, `coverage.mdx`); Polar doesn't say. Compare the logged `rmssd_ms` with `polar_hrv_ms`.
4. **Quality-measurement semantics.** I assume each level holds until the next entry; it could instead describe a window.
5. **`recording_time_delta_ms` reference.** Assumed to count from `test_time`. Also, `test_time` is whole seconds, so up to 1 s of absolute offset shifts all beat timestamps; RR values are unaffected.
6. **Recording length and amplitude scale.** Not documented. Tests use 30–60 s at about 1 mV R amplitude. Thresholds are relative, so scale should not matter; polarity is handled.
7. **Real-signal detector performance.** Unknown until run on real or redacted recordings.

## 5. What the PR would need

1. **Maintainer go-ahead** via the proposal issue (`PROPOSAL_POLAR_RRI_DRAFT.md`, adjusted to the corrected framing: ECG fidelity, not the Resilience Score), per `CONTRIBUTING.md:19-21`. Decisions to ask for:
   - the series-type names and IDs (8, 9);
   - attribution of derived data (3.9);
   - whether to persist derived RMSSD (3.8);
   - the quality encoding;
   - whether `app/algorithms/ecg.py` should be shared across providers (`AGENTS.md:106`).
2. **A real or redacted Elixir wrist-ECG payload**, ideally several including a NO_CONTACT/LOW case. Use it to:
   - add a fixture test;
   - settle the unknowns in section 4 from the logged comparisons;
   - check precision/recall against hand-annotated R peaks. `CONTRIBUTING.md:28-30` asks for real-provider exercise, which needs a Polar Elixir device.
3. **Full backend suite on Docker/testcontainers**, plus a startup run to confirm the seed creates IDs 8 and 9.
4. **AI disclosure** in the PR (tool + model), per `CONTRIBUTING.md:43-50`.

## 6. What was run (exactly)

Environment: macOS, Python 3.14.6, `uv sync` (+ `--group code-quality`). **No Docker**, so the
`db` fixture (testcontainers Postgres) is unavailable. Tests were run with `--noconftest` and test
env vars (`ENV=test`, `SECRET_KEY`, `MASTER_KEY` with the values from `tests/conftest.py`).

- `tests/algorithms/test_ecg_algorithms.py`: **47 passed**.
- `tests/providers/polar/test_polar_wrist_ecg.py`: **13 passed**.
- `test_polar_247.py`, `test_polar_strategy.py`, `test_polar_wrist_ecg.py`, `test_provider_coverage.py`, `tests/schemas/`, `tests/algorithms/`: **361 passed**.
- `test_polar_oauth.py`, `test_polar_workouts.py`, `test_polar_webhook_persistence.py`: **26 errors**, all `fixture 'db' not found` (need Postgres). Not run.
- Whole suite with `--noconftest`, branch vs `main` (temporary worktree):
  - branch: 55 failed / 1276 passed / 1437 errors;
  - `main`: 55 failed / 1216 passed / 1437 errors;
  - the set of failing test IDs is identical, so no regressions without a DB.
  - All 1437 errors are conftest fixtures that are unavailable under `--noconftest`: `db` 955, `client` 466, `fast_password_hashing` 10, `mock_celery_app` 6.
  - The 55 failures need Redis on `localhost:6379` or an S3 bucket config, and fail the same way on `main`.
- `ruff check` + `ruff format` (whole backend): clean.
- `ty check` on changed `app/` files: clean.
- `scripts/export_openapi.py --check`: up to date.
- `scripts/generate_coverage_docs.py`: regenerated (adds two rows).

## Status

- Branch `polar-raw-ecg-prototype` is pushed to the fork: <https://github.com/ViSaReVe/open-wearables/tree/polar-raw-ecg-prototype>. Nothing is pushed to upstream `the-momentum/open-wearables`.
- No PR. Two issues (the RMSSD bug report and the Polar wrist-ECG proposal) are prepared; any PR waits for a go-ahead from the core team, per `CONTRIBUTING.md`.

## Appendix A — benchmark script (section 2 grid)

Run from `backend/` with the test env vars set:

```python
import numpy as np
from app.algorithms.ecg import analyze_ecg
from tests.algorithms.synthetic_ecg import make_rr_series, match_peaks, synthesize_ecg

configs = {
    "clean": dict(noise_mv=0.02),
    "noisy": dict(noise_mv=0.10, emg_mv=0.05, mains_mv=0.05, baseline_mv=0.3),
    "vnoisy": dict(noise_mv=0.20, emg_mv=0.10, mains_mv=0.1, baseline_mv=0.5),
}
for fs in (130.0, 250.0, 500.0):
    for name, cfg in configs.items():
        for extra in ({}, {"polarity": -1.0}, {"t_wave_scale": 2.5}):
            tps = fps = fns = 0
            errs, rm_err = [], []
            for seed in range(10):
                rng = np.random.default_rng(seed)
                rr = make_rr_series(60, mean_rr_ms=rng.uniform(600, 1200), rsa_amplitude_ms=rng.uniform(10, 50),
                                    jitter_ms=rng.uniform(5, 20), rng=rng)
                ecg = synthesize_ecg(rr, fs=fs, rng=rng, **cfg, **extra)
                a = analyze_ecg(ecg.times_ms, ecg.signal_mv)
                tp, fp, fn, e = match_peaks(ecg.r_times_s, np.asarray(a.beat_times_ms) / 1000)
                tps, fps, fns = tps + tp, fps + fp, fns + fn
                errs += list(e)
                rm_err.append(a.rmssd_ms - ecg.true_rmssd_ms)
            print(fs, name, extra, "P", tps / (tps + fps), "R", tps / (tps + fns),
                  "rmssd err mean", np.nanmean(rm_err), "max", np.nanmax(np.abs(rm_err)),
                  "withheld", int(np.isnan(rm_err).sum()))
```
