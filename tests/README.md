# Tests

Run `python -m pytest tests -q` from the project root. The 109 tests cover both the original real-passband experiments and the complex-baseband software transceiver.

| File | Tests | Scope | Key checks |
|---|---:|---|---|
| `test_modulation.py` | 16 | Modulation | Per-component instantaneous-frequency amplitude ratio of 1.0000; regression for incorrect integration normalization |
| `test_demodulation.py` | 15 | Demodulation | Round-trip NRMSE < 0.02; best-fit scale recovers Δf; four demodulation methods compared; complex IQ retains the Q component |
| `test_channel_spectra.py` | 32 | Channel and spectrum | AWGN calibration round-trip error < 0.1 dB; two FFT normalization conventions; Parseval's theorem; agreement with Bessel sidebands |
| `test_metrics.py` | 35 | Metrics | NRMSE, alignment, frequency deviation, SNR, and regressions for metric errors |
| `test_sdr.py` | 11 | Complex-baseband transceiver | Blockwise state, resampling, IQ files and simulated interfaces, broadcast reception, and CLI |
| **Total** | **109** | | |

Dependencies are listed in the root [`requirements.txt`](../requirements.txt). The optional sound-card library `sounddevice` is not required for the core tests.
