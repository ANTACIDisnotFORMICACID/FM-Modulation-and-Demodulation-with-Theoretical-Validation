#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp01 —— 基础链路：调制 -> AWGN -> 希尔伯特解调。

整合早期单独绘图脚本的基础链路实验。

产出：
    results/figures/exp01_basic_chain.png / .pdf
    results/data/exp01_metrics.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm_project import (FMConfig, multitone_message, fm_modulate, awgn,
                        demod_hilbert, valid_slice, nrmse_scaled, best_fit_scale,
                        fft_spectrum, estimate_freq_deviation)
import fm_project.plots as P


def main():
    cfg = FMConfig(
        sample_rate=384000.0,
        carrier_freq=20000.0,
        freq_deviation=4000.0,
        duration=0.02,
        message_freqs=(1000.0, 2000.0),
        message_amps=(1.0, 0.5),
        seed=0,
    )
    print("参数:", json.dumps({k: str(v) for k, v in {
        "fs": cfg.sample_rate, "fc": cfg.carrier_freq, "df": cfg.freq_deviation,
        "cutoff": cfg.effective_cutoff, "beta": cfg.mod_index,
        "carson_B": cfg.carson_bandwidth}.items()}, ensure_ascii=False))
    warn = cfg.warnings()
    print("参数告警:", warn if warn else "无")

    rng = np.random.default_rng(cfg.seed)
    m = multitone_message(cfg.message_freqs, cfg.duration, cfg.sample_rate,
                          amps=cfg.message_amps)
    from fm_project import normalize_peak
    m_n, peak = normalize_peak(m)

    s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                    freq_deviation=cfg.freq_deviation)

    snr_db = 30.0
    r = awgn(s, snr_db, rng)

    d = demod_hilbert(r, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                      cutoff=cfg.effective_cutoff)
    sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)

    err = nrmse_scaled(m_n[sl], d[sl])
    sc = best_fit_scale(m_n[sl], d[sl])
    df_meas = estimate_freq_deviation(s, cfg.sample_rate, cfg.carrier_freq)

    print(f"峰值频偏: 设计 {cfg.freq_deviation:.0f} Hz, 实测 {df_meas:.1f} Hz")
    print(f"解调 NRMSE(增益不变, 切片内) = {err:.6f}")
    print(f"最优定标系数 = {sc:.1f} Hz (设计 df = {cfg.freq_deviation:.1f} Hz)")

    # ---- 绘图 ----
    fig, ax = P.plt.subplots(4, 2, figsize=(13, 10))
    fs = cfg.sample_rate
    P.plot_time(m_n, fs, ax[0, 0], "消息 m(t)（已归一化）")
    x, y = fft_spectrum(m_n, fs, magnitude="db", normalize="amplitude")
    P.plot_spectrum(x, y, ax[0, 1], "消息频谱", xlim=(-5000, 5000))

    P.plot_time(s, fs, ax[1, 0], "FM 调制信号 s(t)")
    x, y = fft_spectrum(s, fs, magnitude="db", normalize="amplitude")
    P.plot_spectrum(x, y, ax[1, 1], "FM 频谱", xlim=(cfg.carrier_freq - 12000,
                                                     cfg.carrier_freq + 12000))

    P.plot_time(r, fs, ax[2, 0], f"接收信号（SNR = {snr_db:.0f} dB）")
    x, y = fft_spectrum(r, fs, magnitude="db", normalize="amplitude")
    P.plot_spectrum(x, y, ax[2, 1], "接收频谱", xlim=(cfg.carrier_freq - 12000,
                                                      cfg.carrier_freq + 12000))

    P.plot_time(d[sl], fs, ax[3, 0], "解调输出（已剔除滤波暂态）")
    P.annotate(ax[3, 0], f"NRMSE = {err:.4f}\nscale = {sc:.0f} Hz (设计 {cfg.freq_deviation:.0f} Hz)",
               loc="upper right")
    x, y = fft_spectrum(d[sl], fs, magnitude="db", normalize="amplitude")
    P.plot_spectrum(x, y, ax[3, 1], "解调输出频谱", xlim=(-5000, 5000))

    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.suptitle("exp01  基础链路：FM 调制 → AWGN → 希尔伯特解调", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png, pdf = P.save(fig, "exp01_basic_chain")
    print("已保存:", png)

    P.ensure_dirs()
    with open(os.path.join(P.DATA_DIR, "exp01_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "sample_rate": cfg.sample_rate, "carrier_freq": cfg.carrier_freq,
            "freq_deviation": cfg.freq_deviation, "cutoff": cfg.effective_cutoff,
            "beta": cfg.mod_index, "carson_bandwidth": cfg.carson_bandwidth,
            "snr_db": snr_db, "message_peak": peak,
            "measured_freq_deviation": df_meas,
            "nrmse_scaled": err, "best_fit_scale": sc,
            "warnings": warn, "cjk_font": P.CJK_OK,
        }, fh, ensure_ascii=False, indent=2)
    print("已保存: results/data/exp01_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
