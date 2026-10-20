"""fm_project —— FM 调制解调算法仿真核心包。

设计原则：
  * 包内不出现 plt.show()、不出现硬编码参数、不出现顶层执行代码；
  * 所有物理量在 docstring 中标注单位；
  * 随机性一律通过 np.random.Generator 注入。
"""

from .config import FMConfig
from .signals import (tone_message, multitone_message, random_bits,
                      bits_to_waveform, normalize_peak)
from .metrics import (nrmse, nrmse_scaled, nrmse_trimmed, best_fit_scale, aligned_nrmse,
                      estimate_freq_deviation, loopback_snr_db, dbfs)
from .filters import lowpass, highpass, transient_samples, valid_slice
from .modulation import fm_modulate, fm_phase, fm_quadrature, fm_modulate_from_bits
from .demodulation import (instantaneous_frequency, demod_hilbert, demod_quadrature,
                           demod_discriminator, demod_downconvert_iq)
from .channel import (signal_power, awgn, measured_snr_db, noise_power_for_snr,
                      add_noise_at_level)
from .spectra import (fft_spectrum, psd_welch, fm_bessel_signal, fm_theoretical_psd)
from . import audioio

__all__ = [
    "FMConfig",
    "tone_message", "multitone_message", "random_bits", "bits_to_waveform", "normalize_peak",
    "nrmse", "nrmse_scaled", "nrmse_trimmed", "best_fit_scale", "aligned_nrmse", "estimate_freq_deviation",
    "loopback_snr_db", "dbfs",
    "lowpass", "highpass", "transient_samples", "valid_slice",
    "fm_modulate", "fm_phase", "fm_quadrature", "fm_modulate_from_bits",
    "instantaneous_frequency", "demod_hilbert", "demod_quadrature",
    "demod_discriminator", "demod_downconvert_iq",
    "signal_power", "awgn", "measured_snr_db", "noise_power_for_snr", "add_noise_at_level",
    "fft_spectrum", "psd_welch", "fm_bessel_signal", "fm_theoretical_psd",
    "audioio",
]

__version__ = "0.1.0"
