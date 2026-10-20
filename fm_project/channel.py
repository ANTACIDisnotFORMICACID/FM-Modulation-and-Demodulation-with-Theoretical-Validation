"""信道模块：AWGN 加噪与 SNR 定标。

统一口径（历史代码中的 add_noise 口径不一致，且实/复噪声混用）：
    SNR_dB = 10*log10( P_signal / P_noise )
其中 P 为平均功率。实数信号加实数噪声；解析/复基带信号加复噪声（复噪声
功率由实部虚部各半构成）。
"""
from __future__ import annotations

import numpy as np


def signal_power(x) -> float:
    """平均功率 P = mean(|x|^2)。"""
    x = np.asarray(x)
    return float(np.mean(np.abs(x) ** 2))


def awgn(x, snr_db, rng: np.random.Generator):
    """对信号加高斯白噪声，使 SNR 达到指定值。

    噪声功率按 **总功率** P_noise = P_signal / 10^(SNR/10) 设定，并平均分配到
    实部与虚部（对实数信号即全部落在实部）。

    Parameters
    ----------
    x : ndarray
        输入信号（实数或复数）。
    snr_db : float
        目标信噪比 [dB]。
    rng : numpy.random.Generator
        随机数发生器，保证可复现。

    Returns
    -------
    ndarray
        与输入同形状、同 dtype 的加噪信号。
    """
    x = np.asarray(x)
    p_sig = signal_power(x)
    if p_sig <= 0:
        raise ValueError("输入信号功率为 0，无法定义 SNR")
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    if np.iscomplexobj(x):
        # 复噪声：实部虚部各占一半功率
        n = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
        n = n * np.sqrt(p_noise / 2.0)
    else:
        n = rng.standard_normal(x.shape) * np.sqrt(p_noise)
    return x + n


def measured_snr_db(reference, noisy) -> float:
    """由"干净信号"与"加噪信号"反算实际 SNR [dB]。

    用于验证 awgn() 的定标是否正确（往返误差应 < 0.1 dB）。
    """
    reference = np.asarray(reference)
    noisy = np.asarray(noisy)
    n = noisy - reference
    p_noise = signal_power(n)
    if p_noise <= 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power(reference) / p_noise))


def noise_power_for_snr(signal, snr_db) -> float:
    """返回为达到指定 SNR 所需的总噪声功率。"""
    return signal_power(signal) / (10.0 ** (snr_db / 10.0))


def add_noise_at_level(x, noise_power, rng: np.random.Generator):
    """按给定的【总】噪声功率加噪（用于复现特定的噪声电平场景）。"""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        n = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
        n = n * np.sqrt(noise_power / 2.0)
    else:
        n = rng.standard_normal(x.shape) * np.sqrt(noise_power)
    return x + n
