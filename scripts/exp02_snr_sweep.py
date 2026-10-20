#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp02 —— 抗噪声性能：SNR 扫描与 FM 门限效应。

扫描输入 SNR，测量输出质量与 FM 门限效应。

要点：窄带 FM 的输出 SNR 与 AM 同阶；FM 的优势来自用带宽换信噪比
      (S/N)_out ~= 3*beta^2*(S/N)_in，且仅在输入 SNR 高于门限时成立。

产出：
    results/figures/exp02_snr_sweep.png / .pdf
    results/data/exp02_snr_sweep.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm_project import (FMConfig, multitone_message, fm_modulate, normalize_peak,
                        demod_hilbert, awgn, nrmse_scaled, valid_slice, signal_power)
import fm_project.plots as P


def demod_snr_pair(m_n, d, sl):
    """由解调输出与参考消息计算输出 SNR [dB]，**同时给出两种参考口径**。

    为什么必须同时给出：文献中的 FM 解调增益公式
        (S/N)_out ~= 3 * beta^2 * (S/N)_in
    对"消息幅度 m"的归一化方式**并不唯一**：
        * 若消息按 **峰值** 定义为 1（m_p = 1），增益读数为 10log10(3*beta^2) + 10log10(m_p^2/m_rms^2)
        * 若消息按 **RMS**  定义为 1，增益读数直接等于 10log10(3*beta^2)
    对单音消息，m_p^2/m_rms^2 = 2，两者**恒定相差 3.01 dB**。

    这个差值纯粹来自定义，不是物理差异；但若不声明口径，读者会以为实测与理论
    差了 3 dB。因此本函数返回两种口径，由调用方明确选择。

    Returns
    -------
    (snr_out_peak_ref_db, snr_out_rms_ref_db, snr_gain_peak_db, snr_gain_rms_db)
    """
    ref = m_n[sl] - np.mean(m_n[sl])
    est = d[sl] - np.mean(d[sl])
    s = float(np.dot(est, ref) / np.dot(ref, ref))
    sig = s * ref                                     # 拟合出的信号分量
    noise = est - sig
    p_n = signal_power(noise)
    if p_n <= 0:
        return np.inf, np.inf, np.inf, np.inf
    # 关键：参考功率必须取【拟合信号分量 sig】的功率，而不是原始消息 m 的功率。
    # 因为本项目的 Hilbert 解调输出带有环路总增益 s（约 Kf），若用原始消息的功率作参考，
    # 会丢掉 s^2 这一因子；s 约为 4000 时输出 SNR 会低估约 72 dB。
    p_peak_ref = float(np.max(np.abs(sig))) ** 2
    p_rms_ref = float(np.mean(sig ** 2))
    out_p = float(10 * np.log10(p_peak_ref / p_n))
    out_r = float(10 * np.log10(p_rms_ref / p_n))
    return out_p, out_r, out_p, out_r


def main():
    base = FMConfig(sample_rate=384000.0, carrier_freq=20000.0,
                    duration=0.05, message_freqs=(1000.0, 2000.0),
                    message_amps=(1.0, 0.5), seed=0)
    m = multitone_message(base.message_freqs, base.duration, base.sample_rate,
                          amps=base.message_amps)
    m_n, _ = normalize_peak(m)

    snrs = np.arange(-10.0, 45.1, 2.5)
    n_mc = 5
    betas = [0.5, 2.0, 5.0]
    rows = []
    curves = {b: [] for b in betas}

    for beta in betas:
        df = beta * base.max_message_freq
        cfg = FMConfig(sample_rate=base.sample_rate, carrier_freq=base.carrier_freq,
                       freq_deviation=df, duration=base.duration,
                       message_freqs=base.message_freqs, message_amps=base.message_amps,
                       seed=base.seed)
        s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                        freq_deviation=cfg.freq_deviation)
        for snr in snrs:
            errs, op, orr = [], [], []
            rng = np.random.default_rng(cfg.seed)
            for _ in range(n_mc):
                r = awgn(s, float(snr), rng)
                d = demod_hilbert(r, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                                  cutoff=cfg.effective_cutoff)
                sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
                errs.append(nrmse_scaled(m_n[sl], d[sl]))
                a, b, _, _ = demod_snr_pair(m_n, d, sl)
                op.append(a)
                orr.append(b)
            op_m, or_m = float(np.mean(op)), float(np.mean(orr))
            curves[beta].append(float(np.mean(errs)))
            rows.append(dict(beta=beta, freq_deviation=df, snr_in_db=float(snr),
                             nrmse=float(np.mean(errs)),
                             snr_out_peak_db=op_m, snr_gain_peak_db=op_m - float(snr),
                             snr_out_rms_db=or_m, snr_gain_rms_db=or_m - float(snr)))
        print(f"beta={beta:>4.1f} 完成（{len(snrs)} 个 SNR 点 x {n_mc} 次）")

    fig, ax = P.plt.subplots(1, 2, figsize=(14, 5.5))
    for beta in betas:
        ax[0].semilogy(snrs, curves[beta], "o-", ms=3, label=f"beta={beta:g}")
    ax[0].set_xlabel("输入 SNR [dB]")
    ax[0].set_ylabel("增益不变 NRMSE")
    ax[0].set_title("解调误差 vs 输入 SNR")
    ax[0].legend(fontsize=8)

    for beta in betas:
        sub = [r for r in rows if r["beta"] == beta]
        xs = [r["snr_in_db"] for r in sub]
        ax[1].plot(xs, [r["snr_gain_rms_db"] for r in sub], "o-", ms=3,
                   label=f"beta={beta:g} (RMS 口径)")
        # 理论上界 = 10log10(3 beta^2)，仅在门限之上成立
        ax[1].axhline(10 * np.log10(3 * beta ** 2), color=f"C{betas.index(beta)}",
                      lw=0.8, ls=":", alpha=0.7)
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("输入 SNR [dB]")
    ax[1].set_ylabel("解调增益 (SNR_out - SNR_in) [dB]")
    ax[1].set_title("FM 解调增益与门限效应（虚线 = 理论上界 10log10(3β²)）")
    ax[1].legend(fontsize=8)

    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle("exp02  FM 抗噪声性能与门限效应", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png, _ = P.save(fig, "exp02_snr_sweep")
    print("已保存:", png)

    P.ensure_dirs()
    csv_path = os.path.join(P.DATA_DIR, "exp02_snr_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("已保存:", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
