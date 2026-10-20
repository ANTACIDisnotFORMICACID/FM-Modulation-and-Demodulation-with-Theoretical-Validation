#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy01 —— 声卡回环：单音 FM 端到端验证。

**本脚本需要音频线到位**。线未到时可用 --selftest 做无硬件自检，
验证信号生成、对齐与解调逻辑本身正确。

用法：
    python scripts/deploy01_loopback_tone.py --selftest          # 无硬件自检
    python scripts/deploy01_loopback_tone.py                     # 真实回环
    python scripts/deploy01_loopback_tone.py --list              # 列出设备
    python scripts/deploy01_loopback_tone.py --out-dev 4 --in-dev 1 --fs 48000

产出：
    results/figures/deploy01_loopback.png / .pdf
    results/data/deploy01_result.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm_project import (FMConfig, tone_message, fm_modulate, demod_hilbert,
                        normalize_peak, nrmse_scaled, best_fit_scale,
                        estimate_freq_deviation, valid_slice, fft_spectrum)
from fm_project import audioio
import fm_project.plots as P


def selftest():
    """无硬件自检：验证信号链本身（内存回环）。"""
    print("=" * 70)
    print("无硬件自检：在内存中完成 调制 -> (理想信道) -> 解调")
    print("=" * 70)
    cfg = FMConfig(sample_rate=44100.0, carrier_freq=10000.0,
                   freq_deviation=2000.0, duration=0.2,
                   message_freqs=(1000.0,), message_amps=(1.0,))
    print("参数告警:", cfg.warnings() or "无")
    m = tone_message(1000.0, cfg.duration, cfg.sample_rate)
    m_n, _ = normalize_peak(m)
    s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                    freq_deviation=cfg.freq_deviation)
    d = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                      cutoff=cfg.effective_cutoff)
    sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
    err = nrmse_scaled(m_n[sl], d[sl])
    sc = best_fit_scale(m_n[sl], d[sl])
    df = estimate_freq_deviation(s, cfg.sample_rate, cfg.carrier_freq)
    print(f"输出占用频段: {cfg.carrier_freq - cfg.freq_deviation:.0f} – "
          f"{cfg.carrier_freq + cfg.freq_deviation:.0f} Hz")
    print(f"峰值频偏: 设计 {cfg.freq_deviation:.0f} Hz, 实测 {df:.1f} Hz")
    print(f"解调 NRMSE = {err:.6f}, best_fit_scale = {sc:.1f}")
    ok = err < 0.05 and abs(df - cfg.freq_deviation) / cfg.freq_deviation < 0.01
    print("\n自检结果:", "PASS" if ok else "FAIL")
    print("\n注意：本自检只验证算法链。真实链路还包含声卡频响、直流阻断、")
    print("      时钟漂移与噪声，必须在音频线到位后用真实回环验证。")
    return 0 if ok else 1


def real_loopback(args):
    print("=" * 70)
    print("声卡回环：单音 FM 端到端")
    print("=" * 70)
    out_dev, in_dev = audioio.resolve_devices(args.out_dev, args.in_dev, args.fs)
    devices = audioio.list_devices()
    print("设备:", devices[out_dev]["name"], "(out) /",
          devices[in_dev]["name"], "(in)")

    cfg = FMConfig(sample_rate=float(args.fs), carrier_freq=10000.0,
                   freq_deviation=2000.0, duration=0.3,
                   message_freqs=(1000.0,), message_amps=(1.0,))
    print("参数告警:", cfg.warnings() or "无")
    m = tone_message(1000.0, cfg.duration, cfg.sample_rate)
    m_n, _ = normalize_peak(m)
    s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                    freq_deviation=cfg.freq_deviation)

    print("开始收发（请确认一分二线已接好，且无啸叫）……")
    tx, rx = audioio.play_and_record(s, cfg.sample_rate,
                                     out_device=out_dev,
                                     in_device=in_dev,
                                     gain=args.gain)
    delay, corr = audioio.align_by_xcorr(tx, rx, cfg.sample_rate)
    print(f"往返时延: {1000*delay/cfg.sample_rate:.2f} ms ({delay} 样本), 相关峰 {corr:.3f}")

    rxm = rx[:, 0] if rx.ndim > 1 else rx
    if delay + len(tx) <= len(rxm):
        seg = rxm[delay:delay + len(tx)]
    else:
        seg = np.concatenate([rxm[delay:], np.zeros(delay + len(tx) - len(rxm))])

    n = min(len(seg), len(s))
    seg, s_ref = seg[:n], s[:n]
    pk = float(np.max(np.abs(seg)))
    print(f"接收电平: 峰值 {pk:.4f} ({20*np.log10(pk+1e-15):+.1f} dBFS)")

    if pk < 1e-3:
        print("\nFAIL: 接收几乎为静音 —— 输出侧或连线有问题。")
        print("   请先运行 python scripts/deploy00b_output_test.py --all-chan 做听测。")
        return 2

    d = demod_hilbert(seg, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                      cutoff=cfg.effective_cutoff)
    sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
    err = nrmse_scaled(m_n[:len(d)][sl], d[sl])
    df = estimate_freq_deviation(seg, cfg.sample_rate, cfg.carrier_freq)
    print(f"实测峰值频偏: {df:.1f} Hz (设计 {cfg.freq_deviation:.0f} Hz)")
    print(f"解调 NRMSE = {err:.6f}")

    fig, ax = P.plt.subplots(2, 2, figsize=(13, 8))
    P.plot_time(tx, cfg.sample_rate, ax[0, 0], "发送波形（含首尾静默）")
    P.plot_time(rxm, cfg.sample_rate, ax[0, 1], "接收波形")
    f, mag = fft_spectrum(seg, cfg.sample_rate, magnitude="db", normalize="amplitude")
    P.plot_spectrum(f, mag, ax[1, 0], "接收频谱",
                    xlim=(cfg.carrier_freq - 6000, cfg.carrier_freq + 6000))
    P.plot_time(d[sl], cfg.sample_rate, ax[1, 1], "解调输出")
    P.annotate(ax[1, 1], f"NRMSE = {err:.4f}\ndf = {df:.0f} Hz")
    fig.suptitle("deploy01  声卡回环：单音 FM 端到端", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png, _ = P.save(fig, "deploy01_loopback")
    print("已保存:", png)

    P.ensure_dirs()
    with open(os.path.join(P.DATA_DIR, "deploy01_result.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(out_dev=out_dev, in_dev=in_dev, fs=args.fs,
                       latency_ms=1000 * delay / cfg.sample_rate,
                       corr_peak=corr, rx_peak_dbfs=20 * np.log10(pk + 1e-15),
                       measured_freq_deviation=df, nrmse=err,
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
                  fh, ensure_ascii=False, indent=2)
    print("已保存: results/data/deploy01_result.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="无硬件自检")
    ap.add_argument("--list", action="store_true", help="列出音频设备")
    ap.add_argument("--out-dev", type=int, default=audioio.DEFAULT_OUT_DEV)
    ap.add_argument("--in-dev", type=int, default=audioio.DEFAULT_IN_DEV)
    ap.add_argument("--fs", type=int, default=audioio.DEFAULT_FS)
    ap.add_argument("--gain", type=float, default=audioio.LOOPBACK_MAX_OUTPUT_PEAK,
                    help="数字输出峰值；直连麦克风口首次测试不得高于 0.05")
    args = ap.parse_args()

    if args.list:
        for d in audioio.list_devices():
            print(f'{d["index"]:>3} {d["name"][:46]:<46} in={d["max_in"]} '
                  f'out={d["max_out"]} sr={d["default_sr"]} {d["api"]}')
        return 0
    if args.selftest:
        return selftest()
    return real_loopback(args)


if __name__ == "__main__":
    sys.exit(main())
