"""Measured, reproducible complex-baseband FM experiments E1–E7."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, correlate, correlation_lags, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm_project.channel import measured_snr_db
from fm_project.demodulation import instantaneous_frequency
from fm_project.sdr import ComplexFMModulator, FMDiscriminator, IQChannel, IntegerDecimator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sdr"
FIG = OUT / "figures"
FS = 240000
FA = 48000
DURATION = 0.20
TRIM = 250


def signal(fs=FS, duration=DURATION):
    t = np.arange(int(fs*duration))/fs
    m = (0.40*np.sin(2*np.pi*700*t) +
         0.18*np.sin(2*np.pi*1700*t) +
         0.08*chirp(t, f0=300, f1=3500, t1=duration))
    return m


def tx(fs=FS, deviation=75000):
    m = signal(fs)
    return m, ComplexFMModulator(fs, deviation).process(m)


def demod_audio(iq, fs=FS, deviation=75000, method="product", cfo_correction=0):
    if method == "legacy_complex_gradient":
        recovered = instantaneous_frequency(iq, fs, analytic=False, detrend=False)
        recovered = (recovered-cfo_correction)/deviation
    else:
        recovered = FMDiscriminator(fs, deviation, method=method,
                                    frequency_offset_hz=cfo_correction).process(iq)
    return IntegerDecimator(fs, fs//FA, cutoff_hz=15000).process(recovered)


def score(reference, estimate, *, align=False):
    n = min(len(reference), len(estimate))
    r = np.asarray(reference[:n])
    e = np.asarray(estimate[:n])
    lag = 0
    if align:
        x = r[TRIM:-TRIM]
        y = e[TRIM:-TRIM]
        corr = correlate(y, x, mode="full", method="fft")
        lags = correlation_lags(len(y), len(x), mode="full")
        valid = abs(lags) <= 20
        lag = int(lags[valid][np.argmax(corr[valid])])
    margin = TRIM+20
    r = r[margin:n-margin]
    e = e[margin+lag:n-margin+lag]
    nmse = float(np.mean((e-r)**2)/np.mean((r-r.mean())**2))
    return {"nmse": nmse, "output_snr_db": float(-10*np.log10(max(nmse, 1e-30))),
            "lag_audio_samples": lag}


def save(name, rows, x, y, xlabel, ylabel, title):
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    if isinstance(y, dict):
        for label, values in y.items():
            ax.plot(x, values, marker="o", label=label)
        ax.legend()
    else:
        ax.plot(x, y, marker="o")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    config = {"iq_sample_rate": FS, "audio_sample_rate": FA,
              "duration_s": DURATION, "trim_audio_samples_per_side": TRIM,
              "seeds": [0, 1, 2, 3, 4],
              "snr_definition": "mean complex IQ power / mean total complex noise power",
              "output_snr_definition": "10log10(var(clean recovered audio)/MSE) after fixed trim; no gain fit"}
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")

    # E1: exact IQ-rate roundtrip across sample rates; no resampling ambiguity.
    e1 = []
    for fs in (120000, 240000, 480000):
        m, z = tx(fs, 30000)
        recovered = FMDiscriminator(fs, 30000).process(z)
        error = recovered[1:]-m[1:]
        e1.append({"iq_rate_hz": fs, "deviation_hz": 30000,
                   "mse": float(np.mean(error**2)),
                   "max_abs_error": float(np.max(abs(error))),
                   "message_rms": float(np.sqrt(np.mean(m*m))),
                   "iq_peak": float(np.max(abs(z)))})
    save("e1_noiseless", e1, [r["iq_rate_hz"] for r in e1],
         [r["max_abs_error"] for r in e1], "IQ rate (Hz)", "Max absolute error",
         "E1: noiseless complex-FM recovery")

    m, z = tx()
    clean = demod_audio(z)
    # E2: output SNR as a function of measured input IQ SNR.
    e2 = []
    for snr in (0, 5, 10, 15, 20, 30):
        vals = []
        actual = []
        for seed in (0, 1, 2):
            noisy = IQChannel(FS, snr_db=snr, seed=seed).process(z)
            vals.append(score(clean, demod_audio(noisy))["output_snr_db"])
            actual.append(measured_snr_db(z, noisy))
        e2.append({"iq_snr_target_db": snr, "iq_snr_measured_mean_db": float(np.mean(actual)),
                   "audio_snr_mean_db": float(np.mean(vals)),
                   "audio_snr_std_db": float(np.std(vals)), "trials": 3})
    save("e2_snr", e2, [r["iq_snr_measured_mean_db"] for r in e2],
         [r["audio_snr_mean_db"] for r in e2], "Measured input IQ SNR (dB)",
         "Output audio SNR (dB)", "E2: complex-IQ noise response")

    # E3: carrier frequency offset, with and without known-offset correction.
    e3 = []
    for cfo in (0, 2000, 5000, 10000, 20000):
        shifted = IQChannel(FS, cfo_hz=cfo).process(z)
        raw = score(clean, demod_audio(shifted))
        corrected = score(clean, demod_audio(shifted, cfo_correction=cfo))
        e3.append({"cfo_hz": cfo, "uncorrected_nmse": raw["nmse"],
                   "corrected_nmse": corrected["nmse"]})
    save("e3_cfo", e3, [r["cfo_hz"] for r in e3],
         {"uncorrected": [r["uncorrected_nmse"] for r in e3],
          "known CFO removed": [r["corrected_nmse"] for r in e3]},
         "Carrier offset (Hz)", "NMSE", "E3: frequency-offset compensation")

    # E4: modulation deviation at a fixed input IQ SNR.
    e4 = []
    for dev in (5000, 15000, 30000, 50000, 75000, 95000):
        _, signal_iq = tx(FS, dev)
        reference = demod_audio(signal_iq, deviation=dev)
        vals = []
        for seed in (0, 1, 2):
            noisy = IQChannel(FS, snr_db=10, seed=seed).process(signal_iq)
            vals.append(score(reference, demod_audio(noisy, deviation=dev))["output_snr_db"])
        e4.append({"deviation_hz": dev, "iq_snr_db": 10,
                   "audio_snr_mean_db": float(np.mean(vals)),
                   "audio_snr_std_db": float(np.std(vals)), "trials": 3})
    save("e4_deviation", e4, [r["deviation_hz"] for r in e4],
         [r["audio_snr_mean_db"] for r in e4], "Peak deviation (Hz)",
         "Output audio SNR (dB)", "E4: FM deviation at 10 dB IQ SNR")

    # E5: two equivalent phase-difference implementations and legacy gradient.
    e5 = []
    noisy = IQChannel(FS, snr_db=15, seed=3).process(z)
    for method in ("product", "unwrap", "legacy_complex_gradient"):
        y = demod_audio(noisy, method=method)
        timings = []
        for _ in range(20):
            start = perf_counter()
            demod_audio(noisy, method=method)
            timings.append((perf_counter()-start)*1000)
        metric = score(clean, y)
        e5.append({"method": method, "audio_snr_db": metric["output_snr_db"],
                   "dsp_elapsed_mean_ms": float(np.mean(timings)),
                   "dsp_elapsed_p95_ms": float(np.percentile(timings, 95))})
    save("e5_methods", e5, list(range(len(e5))),
         {"SNR": [r["audio_snr_db"] for r in e5]},
         "0=product, 1=unwrap, 2=legacy gradient", "Output audio SNR (dB)",
         "E5: discriminator implementation comparison")

    # E6: channel bandwidth; align measured group delay before error scoring.
    e6 = []
    for width in (30000, 60000, 100000, 140000, 180000, 210000):
        filtered = IQChannel(FS, bandwidth_hz=width).process(z)
        metric = score(clean, demod_audio(filtered), align=True)
        e6.append({"channel_bandwidth_hz": width, **metric})
    save("e6_bandwidth", e6, [r["channel_bandwidth_hz"] for r in e6],
         [r["output_snr_db"] for r in e6], "Two-sided channel bandwidth (Hz)",
         "Recovery SNR (dB)", "E6: bandwidth restriction")

    # E7: repeated low-SNR sweep to show the FM threshold region.
    e7 = []
    for snr in (-5, 0, 2, 4, 6, 8, 10, 15, 20):
        vals = []
        for seed in (0, 1, 2, 3, 4):
            noisy = IQChannel(FS, snr_db=snr, seed=seed).process(z)
            vals.append(score(clean, demod_audio(noisy))["output_snr_db"])
        e7.append({"iq_snr_db": snr, "audio_snr_mean_db": float(np.mean(vals)),
                   "audio_snr_std_db": float(np.std(vals)),
                   "audio_snr_min_db": float(np.min(vals)),
                   "audio_snr_max_db": float(np.max(vals)), "trials": 5})
    save("e7_threshold", e7, [r["iq_snr_db"] for r in e7],
         [r["audio_snr_mean_db"] for r in e7], "Input IQ SNR (dB)",
         "Output audio SNR (dB)", "E7: FM noise threshold")

    f, p = welch(z, fs=FS, nperseg=4096, return_onesided=False)
    order = np.argsort(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f[order]/1000, 10*np.log10(np.maximum(p[order], 1e-20)))
    ax.set(title="Complex IQ PSD", xlabel="Offset frequency (kHz)", ylabel="PSD (dB/Hz)")
    fig.tight_layout()
    fig.savefig(FIG / "iq_psd.png", dpi=140)
    plt.close(fig)
    print("E1–E7: measured CSV and figures saved under", OUT)
    print("E7 mean output SNR:", [(r["iq_snr_db"], round(r["audio_snr_mean_db"], 2)) for r in e7])


if __name__ == "__main__":
    main()
