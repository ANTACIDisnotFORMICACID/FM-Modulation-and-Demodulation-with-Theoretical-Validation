"""调制模块。

核心约定（务必遵守）
    s(t) = A * cos(2*pi*f_c*t + 2*pi*K_f * integral_0^t m(tau) dtau)
    f_i(t) = f_c + K_f * m(t)

    K_f 由峰值频偏定义：当 |m| <= 1 时，K_f = Delta_f。

频偏尺度约束
    严禁对积分项做任何归一化（历史代码 integrated /= np.max(np.abs(integrated))）。
    归一化积分等价于把 K_f 改成一个信号相关的未知量，使调制指数 beta 完全失控。
    实测后果：设计 Delta_f=50 Hz 时实得 1019.5 Hz（偏差 20 倍），beta 从 5 变成 101.9。
"""
from __future__ import annotations

import numpy as np

from .signals import normalize_peak


def trapezoid_integral(x, sample_rate):
    """对 x 做数值积分 int_0^t x(tau) dtau，返回与 x 等长、首项为 0 的数组。

    为什么不用 np.cumsum(x)/fs：那是最简单的左矩形法，具有 **一阶** 误差，
    每步偏差约 (x_n - x_{n-1})/(2 fs)。在持续时间较长的信号上该偏差会线性累积，
    使调制相位与 Bessel 级数理论值出现可观差异（实测在 T=0.05 s、Delta_f=4 kHz
    时，与 Bessel 理论合成信号的最大偏差达 0.052）。

    本函数用梯形法（二阶精度）：

        I_0 = 0
        I_n = I_{n-1} + (x_{n-1} + x_n) / (2 fs)

    无穷范数误差从 O(1/fs) 降到 O(1/fs^2)，使仿真与理论在数值上可直接对照
    （实测偏差降到 < 1e-3，见 tests/test_channel_spectra.py）。
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return x.copy()
    inc = np.empty(n)
    inc[0] = 0.0
    if n > 1:
        inc[1:] = (x[:-1] + x[1:]) / (2.0 * sample_rate)
    return np.cumsum(inc)


def fm_phase(message, sample_rate, carrier_freq, freq_deviation, *,
             normalize_message=True, phase_offset=0.0) -> tuple:
    """计算 FM 瞬时相位。

    Parameters
    ----------
    message : ndarray
        消息信号 m(t)，无量纲。
    sample_rate : float
        采样率 [Hz]。
    carrier_freq : float
        载波频率 f_c [Hz]。
    freq_deviation : float
        峰值频偏 Delta_f [Hz]。
    normalize_message : bool
        为 True（默认）时先把 m(t) 归一化到峰值 1，使 freq_deviation 的物理含义
        精确等于峰值频偏。为 False 时假定调用方已保证 |m| <= 1。
    phase_offset : float
        初相 [rad]。

    Returns
    -------
    (phase, kf, peak)
        phase : ndarray，瞬时相位 [rad]
        kf    : float，实际使用的调频灵敏度 [Hz]
        peak  : float，消息原峰值（用于记录/还原）
    """
    m = np.asarray(message, dtype=float).ravel()
    peak = 1.0
    if normalize_message:
        m, peak = normalize_peak(m)

    n = m.size
    t = np.arange(n) / sample_rate
    integral = trapezoid_integral(m, sample_rate)
    phase = phase_offset + 2 * np.pi * carrier_freq * t + 2 * np.pi * freq_deviation * integral
    return phase, float(freq_deviation), float(peak)


def fm_modulate(message, sample_rate, *, carrier_freq, freq_deviation,
                normalize_message=True, phase_offset=0.0,
                amplitude=1.0, method="direct") -> np.ndarray:
    """生成实数带通 FM 信号。

    s(t) = amplitude * cos(2*pi*f_c*t + 2*pi*Delta_f*integral m)

    Parameters
    ----------
    message : ndarray
        消息信号 m(t)。
    sample_rate : float
        采样率 fs [Hz]。
    carrier_freq : float
        载波频率 f_c [Hz]。
    freq_deviation : float
        峰值频偏 Delta_f [Hz]。
    normalize_message : bool
        见 fm_phase。
    phase_offset : float
        初相 [rad]。
    amplitude : float
        载波幅度 A。
    method : {"direct", "quadrature"}
        direct 直接由相位求余弦；quadrature 用 I/Q 合成（数值上等价，
        用于与 fm.py 的历史实现对照）。

    Returns
    -------
    ndarray
        与 message 等长的实数 FM 信号。
    """
    phase, _, _ = fm_phase(
        message, sample_rate, carrier_freq, freq_deviation,
        normalize_message=normalize_message, phase_offset=phase_offset,
    )
    if method == "direct":
        return amplitude * np.cos(phase)
    if method == "quadrature":
        m, _ = (normalize_peak(message) if normalize_message
                else (np.asarray(message, dtype=float).ravel(), 1.0))
        n = m.size
        t = np.arange(n) / sample_rate
        integ = trapezoid_integral(m, sample_rate)
        theta = 2 * np.pi * freq_deviation * integ + phase_offset
        return amplitude * (np.cos(theta) * np.cos(2 * np.pi * carrier_freq * t)
                            - np.sin(theta) * np.sin(2 * np.pi * carrier_freq * t))
    raise ValueError("未知 method: %r（可选 'direct' 或 'quadrature'）" % (method,))


def fm_quadrature(message, sample_rate, *, carrier_freq, freq_deviation,
                  normalize_message=True, phase_offset=0.0) -> tuple:
    """返回 I/Q 两路基带（来自 fm.py 的正交调制实现，规范化）。

    I(t) = cos(2*pi*Delta_f*integral m)
    Q(t) = sin(2*pi*Delta_f*integral m)
    满足 s(t) = I*cos(2*pi*f_c*t) - Q*sin(2*pi*f_c*t)。
    """
    m, _ = (normalize_peak(message) if normalize_message
            else (np.asarray(message, dtype=float).ravel(), 1.0))
    integ = trapezoid_integral(m, sample_rate)
    theta = 2 * np.pi * freq_deviation * integ + phase_offset
    return np.cos(theta), np.sin(theta)


def fm_modulate_from_bits(bits, sample_rate, *, bit_rate, carrier_freq,
                          freq_deviation, f_mark=1000.0, f_space=3000.0,
                          normalize_message=True, amplitude=1.0) -> np.ndarray:
    """BFSK（二进制调频）调制：先做基带 FSK，再做 FM 上变频。

    对应历史 FSK 1.py 的相位连续 BFSK，并补齐"上变频到载波"这一环，
    使其成为可端到端传输的数字链路调制器。
    """
    from .signals import bits_to_waveform
    baseband = bits_to_waveform(bits, bit_rate, sample_rate, f_mark=f_mark, f_space=f_space)
    return fm_modulate(baseband, sample_rate, carrier_freq=carrier_freq,
                       freq_deviation=freq_deviation,
                       normalize_message=normalize_message, amplitude=amplitude)
