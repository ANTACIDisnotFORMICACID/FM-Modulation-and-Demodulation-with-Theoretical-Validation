"""参数集中管理。

原实通带实验的物理量带单位，集中在 dataclass 中。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json


@dataclass
class FMConfig:
    """FM 链路的一组完整参数。

    Attributes
    ----------
    sample_rate : float
        采样率 fs [Hz]。
    carrier_freq : float
        载波频率 f_c [Hz]。
    freq_deviation : float
        峰值频偏 Δf [Hz]。**其物理含义是峰值频偏**，前提是消息已归一化到 |m|<=1
        （见 modulation.fm_modulate 的 normalize_message）。禁止对积分项做归一化。
    duration : float
        时长 T [s]。
    message_freqs : tuple of float
        消息多音频率 [Hz]。
    message_amps : tuple of float
        与之对应的幅度（线性）。
    amplitude : float
        载波幅度 A（线性）。
    cutoff : float or None
        解调后低通截止频率 [Hz]。None 表示自动取 2 x max(message_freqs)。
    seed : int
        随机种子，保证可复现。
    """

    sample_rate: float = 48000.0
    carrier_freq: float = 10000.0
    freq_deviation: float = 2000.0
    duration: float = 0.25
    message_freqs: tuple = (1000.0, 2000.0)
    message_amps: tuple = (1.0, 0.5)
    amplitude: float = 1.0
    cutoff: float | None = None
    seed: int = 0

    # ---- 派生量 ----
    @property
    def n_samples(self) -> int:
        return int(round(self.duration * self.sample_rate))

    @property
    def max_message_freq(self) -> float:
        return float(max(self.message_freqs))

    @property
    def effective_cutoff(self) -> float:
        """解调低通截止频率，缺省为 2 x 消息最高频率。"""
        if self.cutoff is not None:
            return float(self.cutoff)
        return 2.0 * self.max_message_freq

    @property
    def mod_index(self) -> float:
        """调制指数 beta = Delta_f / f_m(max)。"""
        return self.freq_deviation / self.max_message_freq

    @property
    def carson_bandwidth(self) -> float:
        """Carson 带宽估计 B ~= 2(Delta_f + f_m,max) [Hz]。"""
        return 2.0 * (self.freq_deviation + self.max_message_freq)

    # ---- 合法性与告警 ----
    def warnings(self) -> list:
        """返回参数合理性的告警列表（空列表表示无问题）。

        检查采样率、频率、频偏和滤波截止频率的基本约束。
        """
        out = []
        nyq = 0.5 * self.sample_rate
        if self.carrier_freq <= 0:
            out.append(f"载波频率必须为正，当前 {self.carrier_freq} Hz")
        if self.carrier_freq + self.freq_deviation >= nyq:
            out.append(
                f"f_c + Δf = {self.carrier_freq + self.freq_deviation:.1f} Hz 已达/超过奈奎斯特 "
                f"{nyq:.1f} Hz，将发生混叠"
            )
        if self.carrier_freq - self.freq_deviation <= 0:
            out.append(
                f"f_c - Δf = {self.carrier_freq - self.freq_deviation:.1f} Hz <= 0，"
                f"瞬时频率会越过 0 Hz 导致负频折叠"
            )
        if self.effective_cutoff > self.carrier_freq:
            out.append(
                f"解调截止频率 {self.effective_cutoff:.1f} Hz 高于载波频率 "
                f"{self.carrier_freq:.1f} Hz，低通将无法分离消息"
            )
        if self.effective_cutoff <= self.max_message_freq:
            out.append(
                f"解调截止频率 {self.effective_cutoff:.1f} Hz 未高于消息最高频率 "
                f"{self.max_message_freq:.1f} Hz，消息将被削掉"
            )
        return out

    def check(self, raise_on_error: bool = False) -> list:
        w = self.warnings()
        if raise_on_error and w:
            raise ValueError("参数不合法: " + "; ".join(w))
        return w

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent, default=str)
