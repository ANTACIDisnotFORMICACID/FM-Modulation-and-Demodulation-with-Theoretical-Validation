"""消息源与基础信号工具。

随机性通过 Generator 注入，便于复现。
"""
from __future__ import annotations

import numpy as np


def normalize_peak(x: np.ndarray) -> tuple:
    """把信号归一化到峰值 1，返回 (归一化信号, 峰值)。

    峰值 > 0 时返回 x / peak；否则原样返回。
    """
    x = np.asarray(x, dtype=float)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x.copy(), 0.0
    return x / peak, peak


def tone_message(freq, duration, sample_rate, *, amplitude=1.0, phase=0.0) -> np.ndarray:
    """单音消息 m(t) = amplitude * cos(2*pi*freq*t + phase)。

    Parameters
    ----------
    freq : float
        消息频率 [Hz]。
    duration : float
        时长 [s]。
    sample_rate : float
        采样率 [Hz]。
    amplitude : float
        幅度（线性）。
    phase : float
        初相 [rad]。
    """
    n = int(round(duration * sample_rate))
    t = np.arange(n) / sample_rate
    return amplitude * np.cos(2 * np.pi * freq * t + phase)


def multitone_message(freqs, duration, sample_rate, *, amps=None, phases=None) -> np.ndarray:
    """多音叠加消息 m(t) = sum_k a_k cos(2*pi*f_k*t + phi_k)。

    这是历史上 Example 2.py / FM test4.py 所用消息的规范化版本。

    Parameters
    ----------
    freqs : sequence of float
        各音频率 [Hz]。
    duration : float
        时长 [s]。
    sample_rate : float
        采样率 [Hz]。
    amps : sequence of float, optional
        各音幅度；缺省全为 1。
    phases : sequence of float, optional
        各音初相 [rad]；缺省全为 0。
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if amps is None:
        amps = np.ones_like(freqs)
    if phases is None:
        phases = np.zeros_like(freqs)
    n = int(round(duration * sample_rate))
    t = np.arange(n) / sample_rate
    out = np.zeros(n, dtype=float)
    for f, a, p in zip(freqs, amps, phases):
        out += a * np.cos(2 * np.pi * f * t + p)
    return out


def random_bits(n, rng: np.random.Generator) -> np.ndarray:
    """生成 n 个随机的 0/1 比特。"""
    return rng.integers(0, 2, size=int(n)).astype(np.int8)


def bits_to_waveform(bits, bit_rate, sample_rate, *, f_mark=1000.0, f_space=3000.0) -> np.ndarray:
    """把比特流映射为 BFSK 基带波形（相位连续，Manchester 可选由调用方处理）。

    Parameters
    ----------
    bits : sequence of int
        0/1 序列。
    bit_rate : float
        码元率 [bit/s]。
    sample_rate : float
        采样率 [Hz]。
    f_mark : float
        比特 1 对应的基带频率 [Hz]。
    f_space : float
        比特 0 对应的基带频率 [Hz]。
    """
    bits = np.asarray(bits).ravel().astype(int)
    spb = int(round(sample_rate / bit_rate))          # 每码元样本数
    seg = np.empty(spb * len(bits), dtype=float)
    phase = 0.0
    for i, b in enumerate(bits):
        f = f_mark if b == 1 else f_space
        t = np.arange(spb) / sample_rate
        seg[i * spb:(i + 1) * spb] = np.cos(2 * np.pi * f * t + phase)
        phase += 2 * np.pi * f * spb / sample_rate    # 保持相位连续
    return seg
