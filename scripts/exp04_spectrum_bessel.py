#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp04 —— 仿真频谱 vs Bessel 理论谱。

使用 Bessel 展开与 Carson 公式作为独立理论基准，检验仿真结果。

产出：
    results/figures/exp04_spectrum_bessel.png / .pdf
    results/data/exp04_bessel_lines.csv
"""
import csv
import os
import sys

import numpy as np
from scipy.special import jv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm_project import (tone_message, fm_modulate, fft_spectrum, fm_bessel_signal,
                        fm_theoretical_psd, normalize_peak, valid_slice)
import fm_project.plots as P


def coherent_line_amplitude(x, sample_rate, freq):
    """用相干投影提取 freq 处的谱线幅度（精确，不受窗与栅栏效应影响）。"""
    n = len(x)
    t = np.arange(n) / sample_rate
    return 2.0 * abs(np.mean(np.asarray(x) * np.exp(-1j * 2 * np.pi * freq * t)))


def main():
    fs, fc, fm = 384000.0, 20000.0, 1000.0
    betas = [0.5, 2.0, 5.0, 10.0]
    rows = []

    fig, ax = P.plt.subplots(2, 2, figsize=(14, 9))

    for k, beta in enumerate(betas):
        df = beta * fm
        m = tone_message(fm, 0.1, fs)
        s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)

        # 理论：位于 fc + n*fm 的谱线，幅度 |J_n(beta)|
        nmax = int(np.ceil(beta) + 8)
        ns = np.arange(-nmax, nmax + 1)
        theory = np.abs(jv(ns, beta))

        # 实测：相干投影
        meas = np.array([coherent_line_amplitude(s, fs, fc + n * fm) for n in ns])

        # 与 Bessel 级数合成信号互验
        s_bes = fm_bessel_signal(df, fm, fs, 0.1, carrier_freq=fc, amplitude=1.0)
        sl = slice(int(0.02 * len(s)), int(0.98 * len(s)))
        synth_err = float(np.max(np.abs(s[sl] - s_bes[sl])))

        mask = theory > 1e-3
        rel = np.abs(meas[mask] - theory[mask]) / theory[mask]
        rows.append(dict(beta=beta, n_lines=int(mask.sum()),
                         max_rel_err=float(np.max(rel)),
                         mean_rel_err=float(np.mean(rel)),
                         synth_max_abs_diff=synth_err))

        a = ax.ravel()[k]
        a.stem(ns, theory, linefmt="C0-", markerfmt="C0o", basefmt=" ",
               label="Bessel 理论 |J_n(β)|")
        a.plot(ns, meas, "rx", ms=5, label="实测（相干投影）")
        a.set_xlabel("边带阶数 n")
        a.set_ylabel("归一化幅度")
        a.set_title(f"β = {beta:g}   （最大相对误差 {np.max(rel)*100:.2f}%, 合成信号最大偏差 {synth_err:.1e}）")
        a.legend(fontsize=8)
        a.grid(True, alpha=0.3)
        print(f"beta={beta:>5.1f}  谱线数={mask.sum():>3}  理论-实测最大相对误差={np.max(rel)*100:>6.2f}%  "
              f"Bessel 级数合成 vs 直接调制最大偏差={synth_err:.3e}")

    fig.suptitle("exp04  实测频谱 vs Bessel 理论谱（双基准自证）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png, _ = P.save(fig, "exp04_spectrum_bessel")
    print("已保存:", png)

    P.ensure_dirs()
    csv_path = os.path.join(P.DATA_DIR, "exp04_bessel_lines.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("已保存:", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
