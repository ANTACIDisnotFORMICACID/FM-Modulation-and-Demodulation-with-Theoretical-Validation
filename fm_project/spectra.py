"""频谱分析模块。

包含三部分：
  1. fft_spectrum  —— 单边/双边幅度谱（含窗函数增益补偿），来自 how to use fft.py；
  2. psd_welch     —— 功率谱密度估计（Welch），来自 Power Spectral Density.py；
  3. fm_bessel_signal / fm_theoretical_psd —— 由 Bessel 展开给出的 FM 理论谱，
     用于 exp04 的"实测 vs 理论"双基准对照。

FM 单音理论（附录 A）：
    s(t) = A * sum_{n=-inf}^{inf} J_n(beta) * cos(2*pi*(f_c + n*f_m)*t)
其中 beta = Delta_f / f_m。频谱由位于 f_c + n*f_m 的离散谱线构成，第 n 条谱线
的归一化幅度为 |J_n(beta)|。
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv
from scipy.signal import get_window, welch


def _window_array(window, n):
    if window is None or window == "boxcar" or window == "rect":
        return np.ones(n)
    if isinstance(window, str):
        return get_window(window, n)
    w = np.asarray(window, dtype=float)
    if w.size != n:
        raise ValueError("自定义窗长度 %d 与信号长度 %d 不匹配" % (w.size, n))
    return w


def fft_spectrum(x, sample_rate, *, window="hann", onesided=True,
                 magnitude="db", ref=1.0, normalize="power"):
    """计算幅度谱。

    Parameters
    ----------
    x : ndarray
        输入信号（实数或复数）。
    sample_rate : float
        采样率 [Hz]。
    window : str or ndarray
        窗函数名（scipy.signal.get_window）或窗数组。默认 "hann"。
    onesided : bool
        实数信号时是否只返回 0..fs/2（复数信号恒为双边）。
    magnitude : {"db", "linear"}
        返回 dB 还是线性幅度。dB 按 20*log10(A/ref) 计算。
    ref : float
        dB 参考幅度（默认 1.0，即 dBFS 口径）。
    normalize : {"power", "amplitude", "none"}
        谱的归一化口径 —— 这是本函数最容易用错的地方，故做成显式选项：

        * `"power"`（默认）：**功率守恒**口径。补偿窗的能量增益，单边谱对非 DC/Nyquist
          谱线乘 2，使 sum(|X|^2) 在 Parseval 意义下等于信号总功率。
          适合观察"能量落在哪些频率"。
        * `"amplitude"`：**单频谱线幅度**口径。只除以窗的相干增益 mean(w)，
          不做单边加倍。此时正弦分量的谱峰读数等于其真实幅度 A。
          **测量"某条谱线有多强"时必须用这个口径**，否则会读大一倍。
        * `"none"`：不做任何补偿。

    Returns
    -------
    (freq, mag) : (ndarray, ndarray)
        频率轴 [Hz] 与幅度。

    Notes
    -----
    两种口径互不兼容：功率守恒口径下，正弦峰值为 A*sqrt(2)（单边加倍所致）；
    幅度口径下为 A。混用会得到 3 dB 的错觉偏差。
    """
    x = np.asarray(x)
    n = x.size
    w = _window_array(window, n)
    xw = x * w
    W = np.fft.fft(xw)

    # FFT 自带 N 倍缩放，且单边谱需要加倍，故归一化要同时处理三件事。
    # 下面是显式推导（以加窗单音 x=A*cos(2*pi*f0*n/fs) 为例）：
    #   DFT 后峰值 |X| = A*N*sum(w)/2
    #   信号能量 sum(x^2) = A^2*N*sum(w^2)/2
    #
    # 幅度口径：要求峰值读数 = A  ->  除以 sum(w)/2
    # 功率口径（单边）：要求 sum(|W_k|^2) = sum(x^2)。由
    #   sum(x^2) = (1/N)*sum_{k=full}|X_k|^2 = (2/N)*sum_{k>=0, k not DC/Nyq}|X_k|^2
    # 得单边系数 sqrt(2/(N*sum(w^2)))。该口径下正弦峰值读数为 A*sqrt(2)。
    if normalize == "power":
        # 单边功率谱，标准归一化：
        #     |W_k| = sqrt(2) * |FFT_k| / sum(w)     (k 非 DC/Nyquist)
        #     |W_DC| = |FFT_DC| / sum(w)
        #     功率  P = sum(|W_k|^2) / n
        # 此处先统一乘 sqrt(2)/sum(w)，随后在 §"单边截断" 中把 DC/Nyquist 上
        # 多乘的 sqrt(2) 除掉。**这一步不可省略**：否则直流会被多计为 2 倍，
        # 使含直流信号的总功率虚高（实测 0.7+正弦 的情形虚高 +33%）。
        eg = float(np.sum(w ** 2))
        W = W * np.sqrt(2.0) / (n * np.sqrt(eg / n))
    elif normalize == "amplitude":
        # 单频谱线幅度：正弦峰值读数应等于 A。单边谱下 |W_k| = A*sum(w)/2。
        W = W / (float(np.sum(w)) / 2.0)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize 只能是 'power' / 'amplitude' / 'none'")

    if np.isrealobj(x) and onesided:
        half = n // 2 + 1
        W = W[:half].copy()
        if normalize == "power":
            # 撤销 DC 与（偶数 N 时的）Nyquist 谱线上多乘的 sqrt(2)
            W[0] /= np.sqrt(2.0)
            if n % 2 == 0:
                W[-1] /= np.sqrt(2.0)
        elif normalize == "none":
            W[1:-1] *= 2.0
        freq = np.fft.rfftfreq(n, 1.0 / sample_rate)
    else:
        W = np.fft.fftshift(W)
        freq = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))

    mag = np.abs(W)
    if magnitude == "db":
        mag = 20.0 * np.log10(mag / ref + 1e-30)
    elif magnitude != "linear":
        raise ValueError("magnitude 只能是 'db' 或 'linear'")
    return freq, mag


def psd_welch(x, sample_rate, *, nperseg=1024, window="hann", onesided=None,
              scaling="density", db=True, ref=1.0):
    """Welch 功率谱密度估计。

    Parameters
    ----------
    x : ndarray
        输入信号。
    sample_rate : float
        采样率 [Hz]。
    nperseg : int
        每段长度。
    window : str
        窗函数名。
    onesided : bool or None
        实数信号默认单边；复数信号默认双边。
    scaling : {"density", "spectrum"}
        density 返回功率/Hz（默认）；spectrum 返回各谱线功率。
    db : bool
        是否返回 dB。
    ref : float
        幅度参考；内部按 10*log10(P/ref^2) 换算。

    Returns
    -------
    (freq, psd) : (ndarray, ndarray)
    """
    x = np.asarray(x)
    if onesided is None:
        onesided = bool(np.isrealobj(x))
    f, p = welch(x, fs=sample_rate, window=window, nperseg=nperseg,
                 return_onesided=onesided, scaling=scaling)
    if db:
        p = 10.0 * np.log10(p / (ref ** 2) + 1e-30)
    return f, p


def fm_bessel_signal(freq_deviation, message_freq, sample_rate, duration,
                     *, carrier_freq=None, amplitude=1.0, n_harmonics=None):
    """由 Bessel 级数合成单音 FM 信号（用于与直接调制法互验）。

        s(t) = A * sum_{n=-N}^{N} J_n(beta) * cos(2*pi*(f_c + n*f_m)*t)

    Parameters
    ----------
    freq_deviation : float
        峰值频偏 Delta_f [Hz]。
    message_freq : float
        单音消息频率 f_m [Hz]。
    sample_rate : float
        采样率 [Hz]。
    duration : float
        时长 [s]。
    carrier_freq : float, optional
        载波频率 [Hz]。仅在返回时用于标注，不参与合成（级数已把 f_c 含在
        (f_c + n*f_m) 中），因此必须提供。
    amplitude : float
        幅度 A。
    n_harmonics : int, optional
        截断阶数 N。缺省取 ceil(beta) + 10，保证截断误差可忽略。

    Returns
    -------
    ndarray
        合成的实数 FM 信号。

    Notes
    -----
    级数中 n 取遍 -N..N。由于 J_{-n} = (-1)^n J_n 且余弦为偶函数，
    直接累加即可得到正确结果。
    """
    if carrier_freq is None:
        raise ValueError("必须提供 carrier_freq")
    beta = freq_deviation / message_freq
    N = int(n_harmonics) if n_harmonics is not None else int(np.ceil(abs(beta)) + 10)
    n = np.arange(-N, N + 1)
    coeff = jv(n, beta)
    t = np.arange(int(round(duration * sample_rate))) / sample_rate
    s = np.zeros(t.size)
    for ni, ci in zip(n, coeff):
        if ci == 0.0:
            continue
        s += ci * np.cos(2 * np.pi * (carrier_freq + ni * message_freq) * t)
    return amplitude * s


def fm_theoretical_psd(freq_deviation, message_freq, *, carrier_freq,
                       n_harmonics=None, freq_axis=None, center_freq=None,
                       rbw=1.0, db=True, ref=1.0):
    """FM 单音的理论离散谱。

    第 n 条谱线位于 f_c + n*f_m，归一化幅度 |J_n(beta)|，功率占比 J_n(beta)^2
    （因 sum_n J_n^2 = 1，总功率已归一）。

    Parameters
    ----------
    freq_deviation, message_freq : float
        Delta_f、f_m [Hz]。
    carrier_freq : float
        f_c [Hz]。
    n_harmonics : int, optional
        截断阶数，缺省 ceil(beta)+10。
    freq_axis : ndarray, optional
        若提供，则把离散谱线以 rbw 为宽度的高斯核铺到该频率轴上，
        便于与 psd_welch 的结果直接叠图比较。
    center_freq : float, optional
        铺谱时的中心频率；缺省等于 carrier_freq。
    rbw : float
        铺谱的等效分辨率带宽 [Hz]。
    db : bool
        是否返回 dB。

    Returns
    -------
    若未给 freq_axis：(line_freqs, line_powers)
    若给了 freq_axis：(freq_axis, psd_on_axis)
    """
    beta = freq_deviation / message_freq
    N = int(n_harmonics) if n_harmonics is not None else int(np.ceil(abs(beta)) + 10)
    n = np.arange(-N, N + 1)
    line_freqs = carrier_freq + n * message_freq
    line_powers = jv(n, beta) ** 2

    if freq_axis is None:
        if db:
            return line_freqs, 10.0 * np.log10(line_powers / (ref ** 2) + 1e-30)
        return line_freqs, line_powers / (ref ** 2)

    # 用高斯核把离散谱线铺到给定频率轴
    fc = carrier_freq if center_freq is None else center_freq
    f = np.asarray(freq_axis, dtype=float)
    sigma = max(rbw, np.finfo(float).eps)
    out = np.zeros_like(f)
    for fl, pl in zip(line_freqs, line_powers):
        out += pl * np.exp(-0.5 * ((f - fl) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    if db:
        return f, 10.0 * np.log10(out / (ref ** 2) + 1e-30)
    return f, out / (ref ** 2)
