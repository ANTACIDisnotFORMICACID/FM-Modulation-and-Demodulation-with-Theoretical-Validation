# Evolution of the FM experiment files

This directory preserves the standalone scripts that preceded the `fm_project` package. Their file names and formats remain as they were written, so the progression from small experiments to reusable modules is visible.

| Early files | Development step | Current location |
|---|---|---|
| `FM 调制.py`, `Example 2.py`, `FM test 6.py`, `FM test4.py`, `FM test5*.py` | Real-passband FM waveforms, message integration, and repeated parameter variations | `fm_project/modulation.py`, `fm_project/config.py` |
| `FM解调 -test.py`, `hilbert 解调.py`, `FM 解调-test2.py`, `FM test3.py`, `fm.py` | Hilbert-phase, single-path downconversion, and derivative-envelope demodulation trials | `fm_project/demodulation.py` |
| `how to use fft.py`, `Power Spectral Density.py`, `FM 调制-PSD-tip.pdf` | Spectrum and power-spectral-density studies | `fm_project/spectra.py`, `scripts/exp04_spectrum_bessel.py` |
| `IQ save and read test.py`, `IQ save and read with tofile &fromfile.py`, `original_signal IQ.py`, `original.iq`, `output.iq` | Experiments with interleaved integer samples and raw complex samples | `fm_project/iqio.py` |
| `whole test.py` | A standalone end-to-end trial with a different set of sample-rate and frequency choices | `scripts/exp01_basic_chain.py` and the later software transceiver |
| `FSK 1.py` | A separate BFSK experiment | Kept only in this archive |

The early FM scripts copied and modified similar modulation and demodulation functions. The package replaced those copies with named algorithms and explicit parameter and measurement modules. The IQ files here belong to those early trials; the separately documented recording in `data/real/` is the project's external broadcast-IQ example.
