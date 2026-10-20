#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy00b_output_test.py —— 输出通路人耳/示波器可验证测试

用途：当自动测量显示"输出无声"时，用本脚本产生**人耳可听**的音频。
该测试用于人工确认输出通道是否有声，不依赖测量仪器。

用法：
    python scripts/deploy00b_output_test.py --list        # 列出设备
    python scripts/deploy00b_output_test.py --device 4    # 在指定设备上放音
    python scripts/deploy00b_output_test.py --all-chan    # 逐通道测试（定位左右声道）
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
import sounddevice as sd


def list_devices():
    print(f'{"idx":>3} {"name":<50} {"in":>3} {"out":>3} {"sr":>7}  api')
    print("-" * 90)
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        print(f'{i:>3} {d["name"][:50]:<50} {d["max_input_channels"]:>3} '
              f'{d["max_output_channels"]:>3} {int(d["default_samplerate"]):>7}  {api}')


def tone(f, dur, fs, amp=0.05):
    t = np.arange(int(dur * fs)) / fs
    r = int(0.01 * fs)
    env = np.ones(len(t))
    env[:r] = np.linspace(0, 1, r)
    env[-r:] = np.linspace(1, 0, r)
    return (amp * env * np.sin(2 * np.pi * f * t)).astype(np.float32)


def play(dev, fs, chan):
    """播放：中文语音提示无法合成，改用明确的音频模式代替。"""
    nch = int(sd.query_devices(dev)["max_output_channels"])
    if chan >= nch:
        print(f"  设备 {dev} 只有 {nch} 个输出通道，跳过 ch={chan}")
        return False
    print(f"  → 设备 {dev} 通道 {chan}：先 1 s 静音，再 3 次 1 kHz 蜂鸣")
    total = tone(1000, 0.6, fs)
    silence = np.zeros(int(0.6 * fs), dtype=np.float32)
    block = np.concatenate([tone(1000, 0.25, fs), silence,
                            tone(1000, 0.25, fs), silence,
                            tone(1000, 0.25, fs), np.zeros(int(0.5 * fs), dtype=np.float32)])
    sig = np.zeros((len(block), nch), dtype=np.float32)
    sig[:, chan] = block
    try:
        sd.play(sig, samplerate=fs, device=dev, blocking=True)
        return True
    except Exception as e:
        print(f"    失败: {str(e).strip().splitlines()[-1][:70]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--fs", type=int, default=44100)
    ap.add_argument("--all-chan", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return 0

    default_out = int(sd.default.device[1])
    outs = [args.device] if args.device is not None else [default_out]
    print("=" * 70)
    print("仅用于耳机听测：先断开耳机输出到麦克风输入的公对公线。")
    print("保持 Windows 耳机音量在 3%，确认有声后再逐步调整。")
    print("每一轮之间有 1 秒间隔，请记下哪一轮听到了声音。")
    print("=" * 70)
    for dev in outs:
        nch = min(2, int(sd.query_devices(dev)["max_output_channels"]))
        chans = range(nch) if args.all_chan else [0]
        for c in chans:
            play(dev, args.fs, c)
            time.sleep(1.0)
    print()
    print("若全部轮次都听不到：")
    print("  1) 检查 Windows 任务栏音量图标是否静音、音量是否为 0")
    print("  2) 检查一分二转接线是否插到底（TRRS 需插紧）")
    print("  3) 检查公对公线是否把转接线的两个分支对接")
    print("  4) 若转接线的耳机分支本身听不到声音，先单独验证耳机能否出声")
    return 0


if __name__ == "__main__":
    sys.exit(main())
