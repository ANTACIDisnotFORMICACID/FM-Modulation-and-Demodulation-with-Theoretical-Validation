#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp05 —— 四种解调方法横向对比。

统一指标：增益不变 NRMSE、相关系数、频偏恢复比例、耗时。

对比方法：
    demod_hilbert        解析信号 + 相位微分（主方法）
    demod_quadrature     正交下变频 + atan2
    demod_discriminator  经典鉴频器（微分 -> 包络检波 -> 低通）
    demod_downconvert_iq 非正交下变频（历史失败路线，作为反面对照）

产出：
    results/figures/exp05_demod_methods.png / .pdf
    results/data/exp05_demod_methods.csv
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm_project import (FMConfig, multitone_message, fm_modulate, normalize_peak,
                        demod_hilbert, demod_quadrature, demod_discriminator,
                        demod_downconvert_iq, nrmse_trimmed, nrmse_scaled,
                        best_fit_scale, valid_slice, awgn)
import fm_project.plots as P

METHODS = [
    ("hilbert", demod_hilbert),
    ("quadrature", demod_quadrature),
    ("discriminator", demod_discriminator),
    ("downconvert_iq", demod_downconvert_iq),
]


def run_once(cfg, m_n, r, name, func):
    """在给定接收信号上跑一种解调方法，返回耗时与输出。"""
    if name == "hilbert":
        t0 = time.perf_counter()
        d = func(r, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                 cutoff=cfg.effective_cutoff)
        dt = time.perf_counter() - t0
        sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
        return d, sl, dt
    if name == "quadrature":
        c = 2 * cfg.freq_deviation + cfg.max_message_freq
        t0 = time.perf_counter()
        d = func(r, cfg.sample_rate, carrier_freq=cfg.carrier_freq, cutoff=c)
        dt = time.perf_counter() - t0
        return d, valid_slice(len(d), cfg.sample_rate, c), dt
    if name == "discriminator":
        c = cfg.freq_deviation + cfg.max_message_freq
        t0 = time.perf_counter()
        d = func(r, cfg.sample_rate, cutoff=c, amplitude_normalize=True)
        dt = time.perf_counter() - t0
        return d, valid_slice(len(d), cfg.sample_rate, c), dt
    # downconvert_iq
    t0 = time.perf_counter()
    d = func(r, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
             cutoff=cfg.effective_cutoff)
    dt = time.perf_counter() - t0
    return d, valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff), dt


def main():
    cfg = FMConfig(sample_rate=384000.0, carrier_freq=20000.0, freq_deviation=4000.0,
                   duration=0.05, message_freqs=(1000.0, 2000.0),
                   message_amps=(1.0, 0.5), seed=0)
    m = multitone_message(cfg.message_freqs, cfg.duration, cfg.sample_rate,
                          amps=cfg.message_amps)
    m_n, _ = normalize_peak(m)
    s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                    freq_deviation=cfg.freq_deviation)

    snrs = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    n_mc = 5
    rows = []
    curves = {name: [] for name, _ in METHODS}

    for snr in snrs:
        for name, func in METHODS:
            errs, scs, dts = [], [], []
            rng = np.random.default_rng(cfg.seed)
            for _ in range(n_mc):
                r = awgn(s, snr, rng)
                d, sl, dt = run_once(cfg, m_n, r, name, func)
                if name in ("hilbert", "quadrature", "discriminator"):
                    errs.append(nrmse_scaled(m_n[sl], d[sl]))
                    scs.append(best_fit_scale(m_n[sl], d[sl]))
                else:
                    errs.append(nrmse_scaled(m_n[sl], d[sl]))
                    scs.append(best_fit_scale(m_n[sl], d[sl]))
                dts.append(dt)
            e = float(np.mean(errs))
            curves[name].append(e)
            rows.append(dict(snr_db=snr, method=name, nrmse=e,
                             nrmse_std=float(np.std(errs)),
                             best_fit_scale=float(np.mean(scs)),
                             ms=1000 * float(np.mean(dts))))
        print(f"SNR={snr:>5.1f} dB  " + "  ".join(
            f"{n}={curves[n][-1]:.4f}" for n, _ in METHODS))

    fig, ax = P.plt.subplots(1, 2, figsize=(14, 5.5))
    for name, _ in METHODS:
        ax[0].semilogy(snrs, curves[name], "o-", label=name, ms=4)
    ax[0].set_xlabel("SNR [dB]")
    ax[0].set_ylabel("增益不变 NRMSE")
    ax[0].set_title("解调误差 vs 信道 SNR（%d 次平均）" % n_mc)
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    # 计算耗时对比
    names = [n for n, _ in METHODS]
    ts = [np.mean([r["ms"] for r in rows if r["method"] == n and r["snr_db"] == 30.0])
          for n in names]
    ax[1].bar(names, ts, color="C0")
    ax[1].set_ylabel("解调耗时 [ms]")
    ax[1].set_title("计算耗时对比（SNR=30 dB, %d 次平均）" % n_mc)
    for i, t in enumerate(ts):
        ax[1].text(i, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)
    ax[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("exp05  四种 FM 解调方法横向对比", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png, _ = P.save(fig, "exp05_demod_methods")
    print("已保存:", png)

    P.ensure_dirs()
    csv_path = os.path.join(P.DATA_DIR, "exp05_demod_methods.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("已保存:", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
