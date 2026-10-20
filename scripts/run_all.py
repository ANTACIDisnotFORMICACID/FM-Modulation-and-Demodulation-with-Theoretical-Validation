#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py —— 一键复现全部仿真实验（exp01–exp05）。

只运行数值仿真实验，不依赖音频硬件。

用法：
    python scripts/run_all.py            # 全部
    python scripts/run_all.py exp01 exp04  # 指定子集
"""
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

EXPERIMENTS = [
    ("exp01", "exp01_basic_chain.py",      "基础链路：调制→AWGN→希尔伯特解调"),
    ("exp03", "exp03_beta_study.py",       "调制指数 beta 与 Carson 带宽律"),
    ("exp04", "exp04_spectrum_bessel.py",  "实测频谱 vs Bessel 理论谱"),
    ("exp05", "exp05_demod_methods.py",    "四种解调方法横向对比"),
    ("exp02", "exp02_snr_sweep.py",        "抗噪性能与门限效应（蒙特卡洛，较慢）"),
]


def main(argv):
    wanted = [a.lower() for a in argv[1:]]
    todo = [e for e in EXPERIMENTS if not wanted or e[0] in wanted]
    if not todo:
        print("没有匹配的实验。可用:", ", ".join(e[0] for e in EXPERIMENTS))
        return 1

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("=" * 72)
    print("FM Modulation and Demodulation with Theoretical Validation —— 仿真实验一键复现")
    print("=" * 72)
    results = []
    for tag, script, desc in todo:
        print(f"\n[{tag}] {desc}")
        print("-" * 72)
        t0 = time.perf_counter()
        rc = subprocess.run([sys.executable, os.path.join(_HERE, script)],
                            cwd=_ROOT, env=env).returncode
        dt = time.perf_counter() - t0
        results.append((tag, rc, dt))
        print(f"[{tag}] {'完成' if rc == 0 else '失败'}，用时 {dt:.1f} s")

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    total = 0.0
    for tag, rc, dt in results:
        total += dt
        print(f"  {tag}: {'OK  ' if rc == 0 else 'FAIL'}  {dt:7.1f} s")
    print(f"  合计: {total:.1f} s")
    failed = [t for t, rc, _ in results if rc != 0]
    if failed:
        print("失败项:", ", ".join(failed))
        return 1
    print("\n全部通过。产物见 results/figures 与 results/data")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
