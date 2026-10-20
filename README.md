# FM Modulation and Demodulation with Theoretical Validation

A Python project for frequency modulation experiments and an offline, complex-baseband FM software transceiver. It covers continuous-phase IQ generation, configurable channel impairments, demodulation, WAV output, and repeatable measurements. It also decodes a third-party recording of a broadcast FM station without requiring local SDR hardware.

## What is included

- **Software transceiver:** mono WAV or a generated test tone → anti-imaging interpolation → complex FM IQ → simulated channel → phase-difference discriminator → anti-alias decimation → WAV.
- **Offline broadcast receiver:** recorded IQ → channel selection → FM discriminator → audio high-pass and de-emphasis → mono WAV.
- **Channel and measurement tools:** complex AWGN, carrier-frequency offset, initial phase, amplitude scaling, bandwidth limitation, IQ/audio SNR, NMSE, PSD, and processing-time comparisons.
- **Theory experiments:** real-passband FM, four demodulation methods, Bessel sidebands, Carson bandwidth, and noise-threshold studies.

The software chain and experiments are tested locally. A [real IQ recording referenced by PySDR](https://www.pysdr.org/content/rds/) has been processed into a playable, non-silent 4-second WAV. Its programme content has not been assessed by listening. No local RF transmitter or receiver has been tested.

## Quick start

Python 3.13 is the tested environment. Install the dependencies in a virtual environment or an existing Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

The following commands use `python` for brevity; substitute the virtual-environment interpreter if needed.

### Generate and receive complex FM

```powershell
python -m fm_project simulate --output results/sdr/demo_recovered.wav --iq-out data/generated/demo.iq --report results/sdr/demo_report.json
```

With no input WAV, `simulate` generates a 1 kHz + 2 kHz test signal. To process an existing mono or stereo WAV:

```powershell
python -m fm_project simulate --input input.wav --output recovered.wav --iq-out sent.iq --snr-db 15
```

Stereo input is downmixed to mono. IQ output is little-endian `complex64` with an adjacent `.iq.json` metadata file. `--help` lists frequency offset, bandwidth, deviation, block size, and other options. Causal resampling introduces a fixed delay; align the WAV files before sample-by-sample comparison.

### Decode a real broadcast recording

The [sample provenance and SHA-256](data/real/README.md) document identifies a 1,000,000-sample, 250 kHz `complex64` recording associated with a 99.5 MHz station. The recording is downloaded separately; raw IQ and recovered audio are excluded from Git.

```powershell
New-Item -ItemType Directory -Force data/real | Out-Null
Invoke-WebRequest 'https://raw.githubusercontent.com/777arc/498x/master/fm_rds_250k_1Msamples.iq' -OutFile data/real/fm_rds_250k_1Msamples.iq
python -m fm_project receive --input data/real/fm_rds_250k_1Msamples.iq --iq-rate 250000 --format cf32le --center-freq 99500000 --output results/sdr/real_fm_recovered.wav --audio-gain 0.5 --report results/sdr/real_fm_report.json --figure results/figures/real_fm_receiver.png
```

This produces 50 kHz mono audio from four seconds of recorded IQ. The checked-in [receiver report](results/sdr/real_fm_report.json) records the processing parameters and signal levels; the [diagnostic figure](results/figures/real_fm_receiver.png) shows the IQ spectrum, discriminator output, and recovered waveform. `receive` also accepts interleaved signed 8-bit (`cs8`) and little-endian signed 16-bit (`ci16le`) IQ when format and sample rate are specified.

## Reproduce the measurements

```powershell
python scripts/run_all.py
python scripts/run_sdr_experiments.py
python -m pytest tests -q
```

`run_all.py` regenerates the original real-passband experiments (`exp01`–`exp05`). `run_sdr_experiments.py` regenerates complex-baseband experiments E1–E7, with CSV files, configuration, and figures in [results/sdr/](results/sdr/). Random seeds are fixed in the scripts. The test suite currently contains **109 passing tests**.

| Experiment | Recorded result | Interpretation |
|---|---|---|
| E1: noiseless recovery | Maximum message error ≤ 9.33×10⁻¹⁵ at 120/240/480 kHz IQ rates | Phase-difference recovery reaches floating-point precision under these conditions. |
| E2: complex-IQ AWGN | Mean output audio SNR rises from 4.87 dB at 0 dB input IQ SNR to 52.07 dB at 30 dB | Three trials per point; see [CSV](results/sdr/e2_snr.csv). |
| E3: carrier-frequency offset | At 20 kHz offset, uncorrected NMSE is 0.719; correcting the **known** offset reduces it to numerical zero | Blind offset estimation is not implemented. |
| E4: FM deviation | At 10 dB IQ SNR, output SNR rises from 8.27 dB at 5 kHz deviation to 33.84 dB at 95 kHz | This trend applies to the tested rate and message bandwidth. |
| E5: discriminator comparison | Product and unwrapped phase-difference methods both reach 37.10 dB output SNR; the older complex-gradient method reaches 34.38 dB | The first two methods are mathematically equivalent when phase increments do not wrap. |
| E6: channel bandwidth | 30–210 kHz sweep gives 12.26–42.50 dB output SNR | The measured curve is not monotonic at every point; see [CSV](results/sdr/e6_bandwidth.csv). |
| E7: low-SNR sweep | Output SNR changes from 14.06 dB at 4 dB input IQ SNR to 28.80 dB at 8 dB | A threshold-like transition appears in this specific simulation; it is not a universal threshold. |

E1–E7 use a 0.20-second 700 Hz + 1700 Hz + chirp message, normally sampled at 240 kHz IQ and 48 kHz audio. Audio SNR compares noisy reception with the same receiver's clean output after a fixed transient trim; it does not fit a gain factor. See [experiment configuration](results/sdr/experiment_config.json) and the CSV files for the exact conditions. The older experiments and their plots are in [results/data/](results/data/) and [results/figures/](results/figures/).

## Signal model and layout

The complex-baseband modulator uses a continuous phase state across blocks:

```text
phase[n] = phase[n-1] + 2π · deviation · message[n] / iq_rate
iq[n]    = exp(j · phase[n])
```

The receiver estimates each phase increment from `angle(iq[n] · conj(iq[n-1]))`. Input message amplitude has a fixed scale (`|message| ≤ 1`); per-block peak normalization would change the effective frequency deviation. Phase increments must remain within the unambiguous interval to avoid aliasing.

| Path | Role |
|---|---|
| [`fm_project/`](fm_project/) | Reusable DSP, channel, receiver, IQ I/O, and command-line package. The DSP classes retain phase and filter state between blocks. |
| [`scripts/`](scripts/) | Executable experiments that choose parameters, run sweeps, and generate result files. |
| [`tests/`](tests/) | Numerical and interface regression tests. |
| [`results/`](results/) | Checked-in figures, CSV files, and compact reports. |
| [`legacy/`](legacy/) | Earlier exploratory scripts retained for code-history comparison; see their [file evolution](legacy/README.md). |

The project does not implement stereo broadcast decoding, RDS decoding, a live SDR driver, or RF transmission. The included sound-card utilities are separate historical experiments and require `sounddevice` if used. Their results do not validate an RF link.
