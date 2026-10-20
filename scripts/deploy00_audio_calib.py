#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy00_audio_calib.py —— 声卡端口标定

本脚本测量端口频响、回环时延和接收电平，不依赖 fm_project 包：

  端口频响 H(f)：发送 1/3 倍频程多音，回环录回，逐频点测幅度
  往返时延：用互相关定位接收信号相对发送信号的延迟
  电平/信噪比：统计接收峰值、噪底、回环 SNR

用法：
    python scripts/deploy00_audio_calib.py --list
    python scripts/deploy00_audio_calib.py                 # 使用 Windows 当前默认端点

输出：
    results/data/deploy00_audio_calib.json   数值结果
    results/figures/deploy00_audio_loopback.png / .pdf
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import signal as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fm_project import audioio

audioio.require_sd()
sd = audioio.sd

FS_DEFAULT = 48000
TONE_MS = 140          # 每个测试音的时长
GAP_MS = 30            # 音间静音（便于逐段稳态测量）
PEAK_TARGET = 0.05     # -26 dBFS：直连麦克风口的首次测试上限
PILOT_PEAK = 0.001     # -60 dBFS：直连无硬件衰减时的首次探测
PILOT_RX_MAX = 0.10    # 预留约 5 倍扫频增幅，防止输入削顶
PILOT_TONE_SNR_MIN_DB = 6.0  # 1 kHz 分量至少比静默段高 6 dB 才算检出


# --------------------------------------------------------------------------
# 设备枚举
# --------------------------------------------------------------------------
def list_devices() -> None:
    print("hostapis:")
    for i, h in enumerate(sd.query_hostapis()):
        print(f'  {i}: {h["name"]:<22} default_in={h["default_input_device"]} default_out={h["default_output_device"]}')
    print()
    print(f'{"idx":>3} {"name":<50} {"in":>3} {"out":>3} {"sr":>7}  api')
    print("-" * 92)
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        print(f'{i:>3} {d["name"][:50]:<50} {d["max_input_channels"]:>3} {d["max_output_channels"]:>3} '
              f'{int(d["default_samplerate"]):>7}  {api}')


# --------------------------------------------------------------------------
# 测试信号：1/3 倍频程多音 + 首尾基准音
# --------------------------------------------------------------------------
def build_freqs(f_lo=20.0, f_hi=20000.0):
    """1/3 倍频程中心频率，并保证首尾各有一个基准音。"""
    n = int(np.ceil(3 * np.log2(f_hi / f_lo)))
    freqs = f_lo * 2 ** (np.arange(n + 1) / 3.0)
    freqs = freqs[freqs <= f_hi]
    return np.concatenate(([100.0], freqs, [1000.0]))


def build_signal(freqs, fs, amp=PEAK_TARGET):
    """构造多音校准信号，返回信号与每段的样本区间。"""
    seg = int(TONE_MS * fs / 1000)
    gap = int(GAP_MS * fs / 1000)
    lead = np.zeros(int(0.2 * fs), dtype=np.float32)      # 前导静音，用于噪底与延迟定位
    parts = [lead]
    spans = []
    for f in freqs:
        t = np.arange(seg) / fs
        # 淡入淡出，避免起止冲击污染频谱
        env = np.ones(seg, dtype=np.float64)
        r = int(0.01 * fs)
        env[:r] = np.linspace(0, 1, r)
        env[-r:] = np.linspace(1, 0, r)
        tone = (amp * env * np.sin(2 * np.pi * f * t)).astype(np.float32)
        start = sum(len(p) for p in parts)
        parts.append(tone)
        spans.append((f, start + r, start + seg - r))     # 只取稳态段
        parts.append(np.zeros(gap, dtype=np.float32))
    parts.append(np.zeros(int(0.3 * fs), dtype=np.float32))
    return np.concatenate(parts), spans, len(lead)


# --------------------------------------------------------------------------
# 采集
# --------------------------------------------------------------------------
def record_and_play(sig, out_dev, in_dev, fs, in_ch):
    """用独立输入/输出流播放并录音，避免 playrec 的设备组合限制。"""
    tx, rec = audioio.play_and_record(
        sig, fs, out_device=out_dev, in_device=in_dev,
        in_channels=in_ch, gain=1.0,
        max_output_peak=audioio.LOOPBACK_MAX_OUTPUT_PEAK,
    )
    return np.asarray(tx, dtype=np.float64), np.asarray(rec, dtype=np.float64)


def build_pilot(fs, peak=PILOT_PEAK):
    """生成 0.25 s、1 kHz 的低电平安全探测音。"""
    n = int(0.25 * fs)
    t = np.arange(n) / fs
    env = np.ones(n, dtype=np.float64)
    ramp = max(1, int(0.01 * fs))
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return (float(peak) * env * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)


def _tone_amplitude(x, fs, freq=1000.0):
    """用单频复投影估计正弦峰值。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if not x.size:
        return 0.0
    t = np.arange(x.size) / fs
    return float(2.0 * np.abs(np.dot(x, np.exp(-2j * np.pi * freq * t))) / x.size)


def analyze_pilot(rec, fs):
    """在整段录音中滑动搜索 1 kHz，兼容 MME 的较大往返延迟。"""
    mono = rec[:, 0] if np.asarray(rec).ndim > 1 else np.asarray(rec)
    win = max(1, int(0.12 * fs))
    step = max(1, int(0.02 * fs))
    starts = range(0, max(1, len(mono) - win + 1), step)
    amplitudes = np.asarray([
        _tone_amplitude(mono[start:start + win], fs) for start in starts
    ])
    best = int(np.argmax(amplitudes))
    tone_amp = float(amplitudes[best])
    noise_amp = float(np.median(amplitudes))
    tone_snr_db = 20 * np.log10((tone_amp + 1e-20) / (noise_amp + 1e-20))
    return tone_amp, noise_amp, float(tone_snr_db), best * step / fs


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------
def measure(rec, sig, spans, lead, fs):
    """返回 (每频点幅度 dBFS, 延迟样本数, 噪底 dBFS, 峰值 dBFS)。"""
    mono_rec = rec[:, 0] if rec.ndim > 1 else rec
    mono_tx = sig

    # --- 延迟：互相关（在监听信号的前 1 秒内搜索）---
    w = int(0.6 * fs)
    a = mono_tx[:w] - mono_tx[:w].mean()
    b = mono_rec[: w + int(0.5 * fs)] - mono_rec[: w + int(0.5 * fs)].mean()
    corr = sps.correlate(b, a, mode="valid")
    lag = int(np.argmax(np.abs(corr)))

    # --- 噪底：前导静音段 ---
    noise_win = mono_rec[: max(1, int(0.15 * fs))]
    noise_rms = float(np.sqrt(np.mean(noise_win ** 2)) + 1e-20)
    peak = float(np.max(np.abs(mono_rec)) + 1e-20)

    # --- 逐频点：取稳态段的 RMS ---
    mags = []
    for f, s0, s1 in spans:
        lo, hi = s0 + lag, s1 + lag
        if hi <= len(mono_rec):
            seg = mono_rec[lo:hi]
            mags.append(20 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-20))
        else:
            mags.append(float("nan"))
    return np.array(mags), lag, 20 * np.log10(noise_rms), 20 * np.log10(peak)


def main():
    ap = argparse.ArgumentParser(description="声卡端口标定")
    ap.add_argument("--list", action="store_true", help="仅列出音频设备后退出")
    ap.add_argument("--out-dev", type=int, default=None,
                    help="输出设备编号；默认使用 Windows 当前默认输出")
    ap.add_argument("--in-dev", type=int, default=None,
                    help="输入设备编号；默认使用 Windows 当前默认输入")
    ap.add_argument("--fs", type=int, default=FS_DEFAULT)
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--pilot-only", action="store_true",
                    help="仅执行 0.25 s 超低电平探测，不进入完整扫频")
    ap.add_argument("--pilot-level", type=float, default=PILOT_PEAK,
                    help="探测数字峰值；直连首次保持默认 0.001")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return 0

    if args.fs not in (44100, 48000):
        print(f"警告: 采样率 {args.fs} 非常用值，可能被设备拒绝")

    args.out_dev, args.in_dev = audioio.resolve_devices(
        args.out_dev, args.in_dev, args.fs
    )
    if not (0.0 < args.pilot_level <= audioio.LOOPBACK_MAX_OUTPUT_PEAK):
        raise SystemExit(
            f"--pilot-level 必须在 (0, {audioio.LOOPBACK_MAX_OUTPUT_PEAK}] 内"
        )

    freqs = build_freqs()
    sig, spans, lead = build_signal(freqs, args.fs)

    print(f"输出设备 {args.out_dev} / 输入设备 {args.in_dev} / fs={args.fs} / 输入通道={args.in_ch}")
    print(f"测试音 {len(freqs)} 个，{freqs[0]:.0f}–{freqs[-1]:.0f} Hz，总时长 {len(sig)/args.fs:.2f} s")
    print(f"先执行 0.25 s 低电平探测（数字峰值 {args.pilot_level:g}）……")

    _, pilot_rec = record_and_play(
        build_pilot(args.fs, args.pilot_level),
        args.out_dev, args.in_dev, args.fs, args.in_ch
    )
    pilot_peak = float(np.max(np.abs(pilot_rec))) if pilot_rec.size else 0.0
    pilot_db = 20 * np.log10(pilot_peak + 1e-20)
    tone_amp, quiet_tone_amp, tone_snr_db, tone_time_s = analyze_pilot(
        pilot_rec, args.fs
    )
    print(f"探测录入峰值: {pilot_peak:.5f} ({pilot_db:+.1f} dBFS)")
    print(f"1 kHz 分量: {tone_amp:.7f}; 中位同频噪声: {quiet_tone_amp:.7f}; "
          f"增量 {tone_snr_db:+.1f} dB; 出现于 {tone_time_s:.2f} s")
    if pilot_peak >= PILOT_RX_MAX:
        print("NO-GO: 低电平探测已过强；请降低麦克风音量并确认 Boost=0 dB。")
        return 3
    if tone_snr_db < PILOT_TONE_SNR_MIN_DB:
        print("NO-SIGNAL: 未可靠检出 1 kHz 探测音；未执行完整扫频。")
        return 2
    if args.pilot_only:
        print("PILOT PASS: 仅完成超低电平探测，未执行完整扫频。")
        return 0

    print("探测电平通过，开始完整低电平扫频……")

    tx, rec = record_and_play(sig, args.out_dev, args.in_dev, args.fs, args.in_ch)
    prefix = len(tx) - len(sig) - int(1.0 * args.fs)
    spans_tx = [(f, s0 + prefix, s1 + prefix) for f, s0, s1 in spans]
    mags, lag, noise_db, peak_db = measure(
        rec, tx, spans_tx, lead + prefix, args.fs
    )

    latency_ms = 1000.0 * lag / args.fs
    snr_db = peak_db - noise_db

    print()
    print(f"往返时延      : {latency_ms:.2f} ms ({lag} 样本)")
    print(f"接收峰值      : {peak_db:+.1f} dBFS")
    print(f"噪底          : {noise_db:+.1f} dBFS")
    print(f"回环 SNR(峰/底): {snr_db:.1f} dB")
    print()
    print(f'{"freq(Hz)":>9} {"level(dBFS)":>12}')
    for f, m in zip(freqs, mags):
        print(f"{f:>9.1f} {m:>12.1f}")

    # --- 判定 ---
    flat = ~np.isnan(mags)
    ripple = float(np.max(mags[flat]) - np.min(mags[flat])) if flat.any() else float("nan")
    print()
    print(f"全带起伏      : {ripple:.1f} dB")
    print("频响、电平、SNR 和时延需结合设备条件分别评估。")

    # --- 落盘 ---
    os.makedirs(os.path.join(_ROOT, "results", "data"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "results", "figures"), exist_ok=True)
    out = {
        "out_dev": args.out_dev, "in_dev": args.in_dev, "fs": args.fs, "in_ch": args.in_ch,
        "latency_ms": latency_ms, "latency_samples": lag,
        "peak_dbfs": peak_db, "noise_dbfs": noise_db, "loopback_snr_db": snr_db,
        "frequency_hz": [float(f) for f in freqs],
        "level_dbfs": [None if np.isnan(m) else float(m) for m in mags],
        "ripple_db": ripple,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    jpath = os.path.join(_ROOT, "results", "data", "deploy00_audio_calib.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n已保存: {jpath}")

    # --- 画图 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].semilogx(freqs, mags, "o-", ms=4)
        ax[0].axhline(-12, color="r", ls="--", lw=1, label="-12 dBFS")
        ax[0].axhline(-3, color="orange", ls="--", lw=1, label="-3 dBFS")
        ax[0].set_xlabel("Frequency [Hz]"); ax[0].set_ylabel("Level [dBFS]")
        ax[0].set_title("Audio loopback frequency response HH(f)")
        ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
        ax[1].plot(np.arange(len(rec)) / args.fs, rec[:, 0] if rec.ndim > 1 else rec, lw=0.5)
        ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Amplitude")
        ax[1].set_title(f"Recorded loopback waveform (latency {latency_ms:.1f} ms)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(_ROOT, "results", "figures", f"deploy00_audio_loopback.{ext}"), dpi=150)
        print(f"已保存: results/figures/deploy00_audio_loopback.png / .pdf")
    except Exception as e:
        print(f"绘图跳过: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
