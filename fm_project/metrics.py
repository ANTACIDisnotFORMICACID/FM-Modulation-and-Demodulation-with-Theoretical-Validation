"""FM 实验的误差、频偏与信噪比指标。

这些函数用于定量比较波形重建、频偏和信噪比。
"""
from __future__ import annotations

import numpy as np


def nrmse(reference, estimate) -> float:
    """标准 NRMSE = ||est - ref|| / ||ref - mean(ref)||（**要求 estimate 已正确标定**）。

    若 estimate 带有未知增益（解调器输出通常如此），请改用 nrmse_scaled()；
    否则增益因子会直接混入指标结果。

    Parameters
    ----------
    reference, estimate : ndarray
        等长的一维数组。

    Returns
    -------
    float
        无量纲。0 表示完全一致；>1 通常意味着估计被严重缩放或相位颠倒。
    """
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.size != estimate.size:
        raise ValueError("长度不一致: reference=%d, estimate=%d" % (reference.size, estimate.size))
    denom = np.std(reference)
    if denom <= 0:
        raise ValueError("参考信号方差为 0，无法计算 NRMSE")
    return float(np.sqrt(np.mean((estimate - reference) ** 2)) / denom)


def nrmse_trimmed(reference, estimate, sample_rate, cutoff, *, order=4) -> float:
    """先剔除滤波暂态、再计算增益不变 NRMSE —— 比较"需要定标"的解调方法时的规范指标。

    需要预先定标的方法（鉴频器、正交下变频）输出含有恒定增益（例如 2*pi*A），
    且滤波暂态区会带来巨大偏置，因此必须在有效区间内、先定标再比较。
    """
    from .filters import valid_slice as _vs
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    sl = _vs(min(reference.size, estimate.size), sample_rate, cutoff, order)
    return nrmse_scaled(reference[sl], estimate[sl])


def nrmse_scaled(reference, estimate) -> float:
    """增益不变的 NRMSE：先按最小二乘把 estimate 缩放到 reference 再算误差。

    为什么需要它：解调器输出天然带一个总增益（本项目的 Hilbert/正交法输出
    单位为 Hz，约为 K_f；鉴频器还会因实现形式带额外常数因子）。若直接算 NRMSE，
    这个"已知量纲因子"会淹没
    真正的波形畸变，使指标失去意义。因此比较解调质量时应统一使用本函数，
    它回答的问题是"波形形状对不对"，而不是"幅度标定对不对"。

    幅度标定本身由 best_fit_scale() 单独评估（它应恢复 K_f）。
    """
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.size != estimate.size:
        raise ValueError("长度不一致: reference=%d, estimate=%d" % (reference.size, estimate.size))
    r = reference - reference.mean()
    e = estimate - estimate.mean()
    denom = float(np.dot(r, r))
    if denom <= 0:
        raise ValueError("参考信号去均值后方差为 0")
    s = float(np.dot(e, r) / denom)
    resid = e - s * r
    # 分母必须用【已定标】参考的 RMS，即 s * std(r)。
    #
    # 为什么不能用 std(r)：estimate 带有未知增益（本项目 Hilbert 法约 Kf，其他
    # 解调器还可能带额外常数因子），残差 resid 的量纲是"estimate 的量纲"，而
    # std(r) 是 reference 的量纲。两者相除会凭空多出一个约等于总增益的因子。
    # 实测在 Kf ≈ 4e3 的情形下，
    # 真实的 1%% 残差会被报成 41，严重误导判断。
    # 除以 s*std(r) 后，指标成为与增益无关的相对误差，且当 estimate 已正确标定时
    # 退化为标准 NRMSE。合成数据自检（scale=1/100/4000，噪声 0.1%）已确认增益不变性。
    denom_scaled = abs(s) * float(np.sqrt(np.mean(r ** 2)))
    if denom_scaled <= 0:
        raise ValueError("最优定标系数为 0，无法归一化")
    return float(np.sqrt(np.mean(resid ** 2)) / denom_scaled)


def best_fit_scale(reference, estimate) -> float:
    """最小二乘最优缩放系数 s* = <est, ref> / <ref, ref>（去均值后）。

    在与"已知 K_f"的对照实验中，s* 应等于实测的 K_f 恢复比例：
    s* ~= K_hat_f / K_f。据此可给出频偏估计的相对误差。
    """
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.size != estimate.size:
        raise ValueError("长度不一致")
    r = reference - reference.mean()
    e = estimate - estimate.mean()
    denom = float(np.dot(r, r))
    if denom <= 0:
        raise ValueError("参考信号去均值后方差为 0")
    return float(np.dot(e, r) / denom)


def aligned_nrmse(reference, estimate, max_lag=64) -> tuple:
    """在 +-max_lag 内搜索最佳时延后计算 NRMSE。

    用于容忍滤波器群时延与声卡缓冲导致的整体时间偏移。

    Returns
    -------
    (best_nrmse, best_lag, best_scale)
    """
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    n = min(reference.size, estimate.size)
    L = min(int(max_lag), n // 4)
    margin = 2 * L
    if n <= 2 * margin:
        raise ValueError("信号过短，无法做时延搜索")

    best = (float("inf"), 0, 0.0)
    for lag in range(-L, L + 1):
        # 参考窗口固定，估计窗口按 lag 平移；两侧都保持相同长度
        lo, hi = margin, n - margin
        r = reference[lo:hi]
        e = estimate[lo + lag: hi + lag]
        if e.size != r.size:
            continue
        r = r - r.mean()
        e = e - e.mean()
        denom = float(np.dot(r, r))
        if denom <= 0:
            continue
        s = float(np.dot(e, r) / denom)
        denom_scaled = abs(s) * float(np.sqrt(np.mean(r ** 2)))
        if denom_scaled <= 0:
            continue
        err = float(np.sqrt(np.mean((e - s * r) ** 2)) / denom_scaled)
        if err < best[0]:
            best = (err, lag, s)
    return best


def estimate_freq_deviation(signal, sample_rate, carrier_freq, *, trim_frac=0.05) -> float:
    """从带通 FM 信号估计峰值频偏 Delta_f [Hz]。

    通过解析信号的解卷绕相位求瞬时频率，取偏离载波的最大绝对值。
    信号两端 trim_frac 比例被丢弃以避开 Hilbert 变换的边缘效应。
    """
    from scipy.signal import hilbert

    x = np.asarray(signal, dtype=float).ravel()
    trim = int(trim_frac * x.size)
    core = x[trim: x.size - trim] if trim > 0 else x
    phase = np.unwrap(np.angle(hilbert(core)))
    inst_freq = np.gradient(phase, 1.0 / sample_rate) / (2 * np.pi)
    return float(np.max(np.abs(inst_freq - carrier_freq)))


def loopback_snr_db(signal, noise_floor) -> float:
    """由信号与噪底计算 SNR [dB]：SNR = 20*log10(rms(signal)/rms(noise))。"""
    s = np.asarray(signal, dtype=float).ravel()
    n = np.asarray(noise_floor, dtype=float).ravel()
    rms_s = float(np.sqrt(np.mean(s ** 2)))
    rms_n = float(np.sqrt(np.mean(n ** 2)))
    if rms_n <= 0:
        return float("inf")
    return float(20 * np.log10(rms_s / rms_n))


def dbfs(x) -> float:
    """返回 x 的 RMS 电平 [dBFS]。"""
    x = np.asarray(x, dtype=float).ravel()
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-15))
