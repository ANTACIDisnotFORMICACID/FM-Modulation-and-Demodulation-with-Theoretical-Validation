#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp03 —— 调制指数 beta 与 Carson 带宽律。

验证：
  * 仿真估计峰值频偏与设计值对照
  * 实测 90%% / 99%% 占用带宽 vs Carson 估计 B = 2*(Delta_f + f_m)
  * 功率谱的 Bessel 谱线位置

产出：
    results/figures/exp03_beta_study.png / .pdf
    results/data/exp03_beta_study.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm_project import (FMConfig, tone_message, fm_modulate, fft_spectrum,
                        psd_welch, estimate_freq_deviation, nrmse_scaled,
                        demod_hilbert, valid_slice)
import fm_project.plots as P


def occupied_bandwidth(freq, psd_lin, frac):
    """返回包含给定功率比例的最小带宽 [Hz]（以谱重心为中心）。"""
    total = np.sum(psd_lin)
    if total <= 0:
        return float("nan")
    order = np.argsort(psd_lin)[::-1]
    cum = np.cumsum(psd_lin[order])
    k = int(np.searchsorted(cum, frac * total))
    idx = order[: k + 1]
    return float(freq[idx].max() - freq[idx].min())


def main():
    fs, fc, fm = 384000.0, 20000.0, 1000.0
    betas = [0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []

    fig, ax = P.plt.subplots(2, 2, figsize=(13, 9))

    for beta in betas:
        df = beta * fm
        cfg = FMConfig(sample_rate=fs, carrier_freq=fc, freq_deviation=df,
                       duration=0.05, message_freqs=(fm,), message_amps=(1.0,))
        m = tone_message(fm, cfg.duration, fs)
        s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)

        df_meas = estimate_freq_deviation(s, fs, fc)
        f, p = psd_welch(s, fs, nperseg=4096, scaling="density", db=False)
        b90 = occupied_bandwidth(f, p, 0.90)
        b99 = occupied_bandwidth(f, p, 0.99)
        carson = cfg.carson_bandwidth

        # 解调质量
        d = demod_hilbert(s, fs, carrier_freq=fc, cutoff=max(2 * fm, 2000.0))
        from fm_project import normalize_peak
        mn, _ = normalize_peak(m)
        sl = valid_slice(len(d), fs, max(2 * fm, 2000.0))
        err = nrmse_scaled(mn[sl], d[sl])

        rows.append(dict(beta=beta, freq_deviation=df, df_measured=df_meas,
                         df_error_pct=100 * (df_meas - df) / df,
                         carson_bw=carson, bw90=b90, bw99=b99,
                         bw99_over_carson=b99 / carson, demod_nrmse=err))
        print(f"beta={beta:>5.1f}  df(设计/实测)={df:>7.0f}/{df_meas:>7.1f} Hz "
              f"({100*(df_meas-df)/df:+.3f}%)  Carson={carson:>7.0f}  "
              f"B90={b90:>7.0f}  B99={b99:>7.0f}  NRMSE={err:.5f}")

        if beta in (1.0, 5.0):
            fr, mg = fft_spectrum(s, fs, magnitude="db", normalize="amplitude")
            P.plot_spectrum(fr, mg, ax[0, 0] if beta == 1.0 else ax[0, 1],
                            f"FM 频谱  beta = {beta:g}",
                            xlim=(fc - 25000, fc + 25000))

    b = np.array([r["beta"] for r in rows])
    dfm = np.array([r["df_measured"] for r in rows])
    c = np.array([r["carson_bw"] for r in rows])
    b99 = np.array([r["bw99"] for r in rows])
    e = np.array([r["df_error_pct"] for r in rows])

    ax[1, 0].plot(b, e, "o-")
    ax[1, 0].axhline(0, color="k", lw=0.8, ls="--")
    ax[1, 0].set_xlabel("调制指数 beta")
    ax[1, 0].set_ylabel("峰值频偏相对误差 [%]")
    ax[1, 0].set_title("仿真估计峰值频偏 vs 设计值")

    ax[1, 1].plot(b, c, "s-", label="Carson: 2(Δf+f_m)")
    ax[1, 1].plot(b, b99, "o-", label="仿真 99% 占用带宽")
    ax[1, 1].plot(b, b99 / c * c.mean() / c.mean() * 1.0, alpha=0.0)
    ax[1, 1].set_xlabel("调制指数 beta")
    ax[1, 1].set_ylabel("带宽 [Hz]")
    ax[1, 1].set_title("Carson 估计 vs 仿真 99% 占用带宽")
    ax[1, 1].legend()

    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.suptitle("exp03  FM 调制指数 beta 与 Carson 带宽律", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png, _ = P.save(fig, "exp03_beta_study")
    print("已保存:", png)

    P.ensure_dirs()
    csv_path = os.path.join(P.DATA_DIR, "exp03_beta_study.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("已保存:", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
