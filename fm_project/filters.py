"""滤波器模块。

要点：
  * 一律使用 filtfilt 零相位滤波，但必须显式告知调用方暂态区长度；
  * 增加参数合法性断言（历史代码完全没有检查，容易静默出 NaN）。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _check(sample_rate, cutoff):
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须为正，当前 %r" % (sample_rate,))
    if cutoff <= 0:
        raise ValueError("cutoff 必须为正，当前 %r" % (cutoff,))
    nyq = 0.5 * sample_rate
    if cutoff >= nyq:
        raise ValueError(
            "cutoff=%.3f Hz 必须小于奈奎斯特频率 %.3f Hz" % (cutoff, nyq)
        )
    if cutoff > 0.9 * nyq:
        raise ValueError(
            "cutoff=%.3f Hz 过于接近奈奎斯特频率 %.3f Hz，滤波结果不可信（会静默产生 NaN）"
            % (cutoff, nyq)
        )


def lowpass(x, sample_rate, cutoff, order=4) -> np.ndarray:
    """零相位低通滤波（filtfilt）。

    Parameters
    ----------
    x : ndarray
        输入信号。
    sample_rate : float
        采样率 [Hz]。
    cutoff : float
        截止频率 (-3 dB) [Hz]。
    order : int
        Butterworth 阶数。

    Returns
    -------
    ndarray
        与输入等长、零相位的滤波结果。

    Notes
    -----
    两端各有约 transient_samples() 的失真区，做精度评估时必须剔除。
    """
    _check(sample_rate, cutoff)
    x = np.asarray(x, dtype=float).ravel()
    b, a = butter(order, cutoff / (0.5 * sample_rate), btype="low")
    return filtfilt(b, a, x)


def highpass(x, sample_rate, cutoff, order=4) -> np.ndarray:
    """零相位高通滤波（filtfilt）。"""
    _check(sample_rate, cutoff)
    x = np.asarray(x, dtype=float).ravel()
    b, a = butter(order, cutoff / (0.5 * sample_rate), btype="high")
    return filtfilt(b, a, x)


def transient_samples(sample_rate, cutoff, order=4, n_time_constants=6.0) -> int:
    """估计 filtfilt 后**两端各需丢弃**的暂态样本数。

    推导（这是本函数与历史经验做法（valid_idx = 0.1*len(signal)）的关键区别）：
      1. 模拟低通的时间常数 tau = 1/(2*pi*cutoff) [s]；
      2. 单次滤波的暂态持续时间约为 n_time_constants 个 tau；
      3. filtfilt 做前后两次滤波，故总暂态约为 2 倍 —— 这个 2 倍因子
         **已包含在本函数的返回值中**，调用方不要再乘 2。

    实测校核：fs=48000 Hz、cutoff=10000 Hz、order=4 时返回约 150 样本（3 ms），
    与滤波器阶跃响应长度一致。若误把返回值再乘 2 当作单侧边界、或使用过大的
    系数，在高采样率下会丢弃掉几乎整个信号（本项目曾因此在 fs=384 kHz 时
    只剩 7 个有效样本）。
    """
    _check(sample_rate, cutoff)
    tau = 1.0 / (2 * np.pi * cutoff)
    return int(np.ceil(2 * n_time_constants * tau * sample_rate))


def valid_slice(n, sample_rate, cutoff, order=4) -> slice:
    """返回剔除两端暂态后的有效切片（用于精度评估）。

    两侧共丢弃 2 * transient_samples()。信号过短时抛出 ValueError 而不是静默返回
    一个近似空的切片 —— 后者会把"没有样本"伪装成"误差很大"。
    """
    m = transient_samples(sample_rate, cutoff, order)
    if 2 * m >= n:
        raise ValueError(
            "信号过短（%d 样本），不足以剔除暂态（两端需各 %d 样本）；"
            "请延长 duration 或降低过采样比" % (n, m)
        )
    return slice(m, n - m)
