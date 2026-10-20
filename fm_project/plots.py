"""统一绘图工具。

绘图约定：
  * 全项目只在此处设置字体与 savefig 口径；
  * 一律 savefig，绝不 plt.show()（脚本运行在无显示环境）；
  * 中文字体缺失时自动回退，保证不出现方框。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_ROOT, "results", "figures")
DATA_DIR = os.path.join(_ROOT, "results", "data")

_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                   "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei"]


def use_cjk_font():
    """选用一个可用的中文字体；找不到则保持默认并返回 False。"""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams.get("font.sans-serif", [])
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK_OK = use_cjk_font()
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.unicode_minus": False,
    "font.size": 9,
})


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def save(fig, name):
    """保存图为 png + pdf 两种格式，返回 (png路径, pdf路径)。"""
    ensure_dirs()
    png = os.path.join(FIG_DIR, name + ".png")
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_time(x, sample_rate, ax=None, title="", ylabel="Amplitude", **kw):
    """时域曲线。"""
    ax = ax or plt.gca()
    t = range(len(x)) if sample_rate is None else None
    import numpy as np
    t = np.arange(len(x)) / sample_rate * 1e3 if sample_rate else np.arange(len(x))
    ax.plot(t, x, lw=0.8, **kw)
    ax.set_title(title)
    ax.set_xlabel("Time [ms]" if sample_rate else "Sample")
    ax.set_ylabel(ylabel)
    return ax


def plot_spectrum(freq, mag_db, ax=None, title="", xlabel="Frequency [kHz]",
                  xlim=None, **kw):
    """频域曲线，频率轴以 kHz 显示。"""
    import numpy as np
    ax = ax or plt.gca()
    ax.plot(np.asarray(freq) / 1e3, mag_db, lw=0.8, **kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Magnitude [dB]")
    if xlim is not None:
        ax.set_xlim(np.asarray(xlim) / 1e3)
    return ax


def annotate(ax, text, loc="upper right", **kw):
    ax.text(0.98 if "right" in loc else 0.02,
            0.95 if "upper" in loc else 0.05,
            text, transform=ax.transAxes,
            ha="right" if "right" in loc else "left",
            va="top" if "upper" in loc else "bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8), **kw)
    return ax
