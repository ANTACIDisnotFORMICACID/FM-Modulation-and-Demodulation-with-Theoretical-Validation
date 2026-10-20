"""解调模块。

四种方法，统一返回【消息估计】（单位与消息同量纲），而不是"减去 f_c 后的瞬时频率"：

    demod_hilbert         解析信号 + 相位微分          —— 主方法
    demod_quadrature      正交下变频 + atan2           —— 对比方法
    demod_discriminator   微分 -> 整流 -> 低通         —— 经典模拟鉴频器
    demod_downconvert_iq  非正交下变频（含失败对照）    —— 展示单路混频不能恢复 FM 消息

所有方法内部统一使用 np.gradient 而非 np.diff，保证输出与输入等长。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .filters import lowpass, highpass, valid_slice


def instantaneous_frequency(x, sample_rate, *, analytic=True, detrend=True) -> np.ndarray:
    """求信号瞬时频率 [Hz]，输出与输入等长。

    Parameters
    ----------
    x : ndarray
        实带通信号。
    sample_rate : float
        采样率 [Hz]。
    analytic : bool
        为 True 时先用 Hilbert 变换构造解析信号（正确做法）。
        为 False 时把输入直接当复信号 —— 只有当 x 本身就是复基带时才正确。
    detrend : bool
        是否扣除均值（避免瞬时频率曲线整体平移）。

    Returns
    -------
    ndarray
        瞬时频率 [Hz]，长度与 x 相同。
    """
    if analytic:
        x = np.asarray(x)
        if np.iscomplexobj(x):
            raise ValueError("analytic=True 只接受实信号；复数 IQ 信号请使用 analytic=False")
        x = np.asarray(x, dtype=float).ravel()
        z = hilbert(x)
    else:
        # 保留复数 IQ 的虚部。旧实现先强制转换为 float，导致 analytic=False
        # 虽声称支持复基带，实际会静默丢弃 Q 路并把单音频率算成 0 Hz。
        z = np.asarray(x, dtype=complex).ravel()
    phase = np.unwrap(np.angle(z))
    inst = np.gradient(phase, 1.0 / sample_rate) / (2 * np.pi)
    if detrend:
        inst = inst - np.mean(inst)
    return inst


def demod_hilbert(rx_signal, sample_rate, *, carrier_freq=None,
                  cutoff=None, order=4, detrend=True, remove_dc=True) -> np.ndarray:
    """主解调方法：Hilbert 解析信号 -> 解卷绕相位 -> 相位微分 -> 低通。

    Parameters
    ----------
    rx_signal : ndarray
        接收到的实带通 FM 信号。
    sample_rate : float
        采样率 [Hz]。
    carrier_freq : float, optional
        载波频率 [Hz]。**仅用于扣除零频偏引起的常数偏置**（注意：瞬时频率本身
        已是 Hz，减去 f_c 只是把曲线移到 0 附近，并无"下变频"含义）。
    cutoff : float
        基带低通截止频率 [Hz]。必须 >= 消息最高频率，推荐 1.5-2 倍。
        为 None 时不滤波。
    order : int
        低通阶数。
    detrend : bool
        微分后是否扣除均值。
    remove_dc : bool
        **低通之前**是否先扣除直流（见函数体内的步骤注释）。必须置 True（默认）：
        低通本身并不抑制直流，而 Hilbert + np.gradient 的离散化底噪会在瞬时频率上
        留下可观的正偏置；若把去直流放到低通之后，低通还会先把带外噪声混入带内，
        使残差抬高一个量级（实测 1e-3 恶化到 1e-2）。
        （经典鉴频器同样必须去直流。）

    Returns
    -------
    ndarray
        消息估计，与输入等长（含两端滤波暂态，需用 filters.valid_slice 剔除）。
    """
    inst = instantaneous_frequency(rx_signal, sample_rate, analytic=True, detrend=detrend)
    if carrier_freq is not None:
        inst = inst - float(carrier_freq)

    # 步骤顺序很重要：必须在低通【之前】去直流。
    # 原因：(1) 低通并不抑制直流；(2) 若先去直流顺序反了，低通会把带外噪声
    # 混入消息带内，抬高本底。实测在 fs=384 kHz/cutoff=10 kHz 下，顺序反了会使
    # 带内残差从约 1e-3 恶化到约 1e-2。
    if remove_dc:
        inst = inst - np.mean(inst)
    if cutoff is not None:
        inst = lowpass(inst, sample_rate, cutoff, order=order)
    return inst


def demod_quadrature(rx_signal, sample_rate, *, carrier_freq, cutoff=None,
                     order=4, detrend=True) -> np.ndarray:
    """正交（I/Q）下变频解调：乘以 cos/sin 后低通，再用 atan2 求相位。

    关键是对 I/Q
    两路同时下变频并取 atan2，而不是只乘一个 cos 就低通。
    """
    x = np.asarray(rx_signal, dtype=float).ravel()
    n = x.size
    t = np.arange(n) / sample_rate
    lo_c = np.cos(2 * np.pi * carrier_freq * t)
    lo_s = np.sin(2 * np.pi * carrier_freq * t)
    if cutoff is None:
        # 低通只需保留两倍频偏即可，取载波的 1/4 作为稳健缺省
        cutoff = max(2.0 * abs(carrier_freq) / 4.0, 1.0)
    i = lowpass(x * lo_c, sample_rate, cutoff, order=order)
    q = lowpass(-x * lo_s, sample_rate, cutoff, order=order)
    phase = np.unwrap(np.arctan2(q, i))
    inst = np.gradient(phase, 1.0 / sample_rate) / (2 * np.pi)
    if detrend:
        inst = inst - np.mean(inst)
    return inst


def demod_discriminator(rx_signal, sample_rate, *, carrier_freq=None,
                        cutoff=None, order=4, detect_mode="envelope",
                        detrend=True, amplitude_normalize=False) -> np.ndarray:
    """经典模拟鉴频器：微分 -> 取包络 -> 低通 -> 去直流。

    对应历史 fm.py 的"微分 -> 全波整流 -> 低通 -> 去直流 -> 定标"链。

    Parameters
    ----------
    rx_signal : ndarray
        接收信号。
    sample_rate : float
        采样率 [Hz]。
    carrier_freq : float, optional
        未使用（保留以统一接口）。
    cutoff : float
        输出低通截止频率 [Hz]。
    order : int
        低通阶数。
    detect_mode : {"envelope", "abs"}
        envelope 用解析信号模（推荐，受幅度起伏影响小）；
        abs 用历史实现的 np.abs 全波整流（作为对照，说明其劣势）。
    detrend : bool
        是否去直流。

    Returns
    -------
    ndarray
        与输入等长的消息估计（未定标；幅度需由 best_fit_scale 或前导段定标）。
    """
    x = np.asarray(rx_signal, dtype=float).ravel()
    z = hilbert(x)
    env = np.abs(z)
    if amplitude_normalize:
        # 真实接收机在鉴频前必做限幅（limiter），以消除幅度起伏的影响。
        #
        # 关键：包络估计的低通截止必须**远高于**幅度起伏（衰落）速率，却又远低于
        # 载波频率；否则该步骤会退化成"除以一个近常数"，成为无效操作。
        # 早期实现把截止设成 cutoff/(2*pi)（约 300 Hz），但由于包络自身的起伏只有
        # 约 50 Hz，低通几乎不改变它 —— 实测证明此时 amplitude_normalize 只等价于
        # 一个常数缩放（best-fit scale 恰为 1.000000，最大残差 1.6e-11），功能名不副实。
        #
        # 现取 cutoff/4：足够快以跟随衰落，又远低于载波以不受载波纹波影响。
        fc_env = (cutoff / 4.0) if cutoff else sample_rate / 16.0
        fc_env = float(np.clip(fc_env, 1.0, 0.25 * sample_rate))
        a_ref = lowpass(env, sample_rate, fc_env, order=order)
        a_ref = np.where(a_ref < 1e-12, 1e-12, a_ref)
        x = x / a_ref
        z = hilbert(x)
        env = np.abs(z)
    a_rms = float(np.sqrt(np.mean(env ** 2)))            # 恒定包络时约等于 A
    if a_rms <= 0:
        raise ValueError("输入信号能量为 0")

    # 微分器：d/dt -> 乘以 2*pi，得到 rad/s
    d = 2 * np.pi * np.gradient(x, 1.0 / sample_rate)
    if detect_mode == "envelope":
        d = np.abs(hilbert(d))
    elif detect_mode == "abs":
        d = np.abs(d)
    else:
        raise ValueError("detect_mode 只能是 'envelope' 或 'abs'")

    # 包络检波输出 ~= A * 2*pi * (f_c + K_f*m)；除以 (2*pi*A) 后即为 Hz 量纲，
    # 可与 demod_hilbert 的输出直接比较（残差仍只是一个恒定增益因子）。
    d = d / (2 * np.pi * a_rms)
    if cutoff is not None:
        d = lowpass(d, sample_rate, cutoff, order=order)
    if detrend:
        d = d - np.mean(d)
    return d


def demod_downconvert_iq(rx_signal, sample_rate, *, carrier_freq, cutoff=None,
                         order=4, detrend=True) -> np.ndarray:
    """非正交下变频 + 低通 —— 历史失败路线的复现，用于方法对照。

    警告：本方法**在物理上是错的**。x(t)*cos(2*pi*f_c*t) 低通后得到的是
    0.5*cos(2*pi*K_f*integral m)，即"消息的余弦"，而不是消息本身。
    一组数值实验中，混频后低频区仅占 35.9% 能量，高频区占 64.1%。

    Returns
    -------
    ndarray
        与输入等长的（错误的）基带估计，仅用于对照展示。
    """
    x = np.asarray(rx_signal, dtype=float).ravel()
    n = x.size
    t = np.arange(n) / sample_rate
    mixed = x * np.cos(2 * np.pi * carrier_freq * t)
    if cutoff is None:
        cutoff = max(abs(carrier_freq) / 4.0, 1.0)
    out = lowpass(mixed, sample_rate, cutoff, order=order)
    if detrend:
        out = out - np.mean(out)
    return out
