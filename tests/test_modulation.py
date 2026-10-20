"""调制正确性测试，验证瞬时频率和频偏约定。

核心断言：瞬时频率必须严格等于 f_c + K_f * m(t)。
"""
import numpy as np
import pytest

from fm_project import (FMConfig, fm_modulate, fm_quadrature, fm_phase,
                        multitone_message, estimate_freq_deviation,
                        normalize_peak, valid_slice)


def make_message(fs=200000.0, duration=0.05, freqs=(1000.0, 2000.0)):
    return multitone_message(freqs, duration, fs, amps=(1.0, 0.5))


class TestInstantaneousFrequency:
    """调制必须使 f_i(t) = f_c + K_f*m(t)，且不得归一化积分项。"""

    def test_instantaneous_frequency_matches_theory(self):
        fs, fc, df = 200000.0, 30000.0, 5000.0
        m = make_message(fs, 0.05)
        m_norm, peak = normalize_peak(m)
        s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)

        from scipy.signal import hilbert
        idx = slice(2000, len(s) - 2000)
        inst = np.gradient(np.unwrap(np.angle(hilbert(s))), 1 / fs)[idx] / (2 * np.pi)
        expected = fc + df * m_norm[idx]

        # 相对载波的偏差才是关键量（绝对误差含相位解卷绕的数值底噪）
        err = inst - expected
        assert np.max(np.abs(err)) < 0.02 * df, (
            "瞬时频率与理论值偏差 %.3f Hz，超过峰偏的 2%%（峰偏 %.1f Hz）"
            % (np.max(np.abs(err)), df)
        )

    def test_no_integral_normalization_regression(self):
        """回归测试：若有人重新引入"归一化积分"的错误，本测试必须失败。"""
        fs, fc, df = 200000.0, 30000.0, 5000.0
        m = make_message(fs, 0.05)

        # 正确实现
        s_ok = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)
        # 复现积分项归一化的历史错误实现
        integ = np.cumsum(m) / fs
        integ_bad = integ / np.max(np.abs(integ))
        t = np.arange(len(m)) / fs
        s_bad = np.cos(2 * np.pi * fc * t + 2 * np.pi * df * integ_bad)

        df_ok = estimate_freq_deviation(s_ok, fs, fc)
        df_bad = estimate_freq_deviation(s_bad, fs, fc)

        # 容差 1e-3：离散 cumsum 积分与 hilbert 数值微分本身带来的残差约 5e-4 量级
        assert abs(df_ok - df) / df < 1e-3, (
            "正确实现的峰值频偏应等于设计值（实测 %.3f Hz / 设计 %.1f Hz）" % (df_ok, df)
        )
        # 错误实现的频偏会被积分项峰值的倒数缩放，通常远偏离设计值
        assert abs(df_bad - df) / df > 0.05, "错误实现应当偏离设计值（本测试用于确认它确实会偏离）"

    def test_freq_deviation_scales_uniformly(self):
        """m 与 3m 应给出相同的峰值频偏（因为归一化消息）。"""
        fs, fc, df = 200000.0, 30000.0, 5000.0
        m = make_message(fs, 0.04)
        d1 = estimate_freq_deviation(fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df), fs, fc)
        d3 = estimate_freq_deviation(fm_modulate(3 * m, fs, carrier_freq=fc, freq_deviation=df), fs, fc)
        assert abs(d1 - d3) < 0.01 * df

    def test_modulation_index_is_exact(self):
        """beta = Delta_f / f_m,max 必须精确可控（历史代码给出 101.9 而非 5）。"""
        cfg = FMConfig(sample_rate=200000.0, carrier_freq=30000.0,
                       freq_deviation=5000.0, duration=0.05,
                       message_freqs=(1000.0,), message_amps=(1.0,))
        assert cfg.mod_index == pytest.approx(5.0)
        m = multitone_message(cfg.message_freqs, cfg.duration, cfg.sample_rate,
                              amps=cfg.message_amps)
        s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                        freq_deviation=cfg.freq_deviation)
        df_meas = estimate_freq_deviation(s, cfg.sample_rate, cfg.carrier_freq)
        assert df_meas == pytest.approx(cfg.freq_deviation, rel=1e-3)

    def test_carson_bandwidth_formula(self):
        cfg = FMConfig(freq_deviation=5000.0, message_freqs=(1000.0, 2000.0))
        assert cfg.carson_bandwidth == pytest.approx(2 * (5000.0 + 2000.0))


class TestQuadratureEquivalence:
    def test_quadrature_equals_direct(self):
        fs, fc, df = 100000.0, 15000.0, 3000.0
        m = make_message(fs, 0.03)
        s_direct = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df, method="direct")
        s_quad = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df, method="quadrature")
        assert np.max(np.abs(s_direct - s_quad)) < 1e-12

    def test_iq_reconstructs_signal(self):
        fs, fc, df = 100000.0, 15000.0, 3000.0
        m = make_message(fs, 0.03)
        I, Q = fm_quadrature(m, fs, carrier_freq=fc, freq_deviation=df)
        t = np.arange(len(m)) / fs
        s = I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t)
        s_ref = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)
        assert np.max(np.abs(s - s_ref)) < 1e-12


class TestConfigWarnings:
    def test_detects_negative_frequency_folding(self):
        """检查瞬时频率越过零点：Kf*max|m| > f_c。"""
        cfg = FMConfig(sample_rate=10000.0, carrier_freq=100.0,
                       freq_deviation=500.0, message_freqs=(10.0,))
        w = cfg.warnings()
        assert any("越过 0 Hz" in x for x in w), "应检出瞬时频率越过 0 Hz"

    def test_detects_aliasing(self):
        cfg = FMConfig(sample_rate=10000.0, carrier_freq=4800.0, freq_deviation=500.0)
        assert any("奈奎斯特" in x for x in cfg.warnings())

    def test_detects_cutoff_below_message(self):
        """截止频率取在消息最高频率处会截断信号。"""
        cfg = FMConfig(message_freqs=(1000.0, 10000.0), cutoff=10000.0)
        assert any("未高于消息最高频率" in x for x in cfg.warnings())

    def test_clean_config_has_no_warnings(self):
        cfg = FMConfig(sample_rate=48000.0, carrier_freq=10000.0,
                       freq_deviation=2000.0, message_freqs=(1000.0, 2000.0))
        assert cfg.warnings() == []

    def test_check_raises_on_error(self):
        cfg = FMConfig(sample_rate=10000.0, carrier_freq=100.0, freq_deviation=500.0)
        with pytest.raises(ValueError):
            cfg.check(raise_on_error=True)


class TestFMPhaseShape:
    def test_output_length_matches_message(self):
        m = make_message(48000.0, 0.02)
        s = fm_modulate(m, 48000.0, carrier_freq=8000.0, freq_deviation=1000.0)
        assert s.shape == m.shape

    def test_amplitude_is_preserved(self):
        """注意：峰值只会在相位恰好落在采样点时达到 A，故用 1e-3 相对容差。"""
        m = make_message(48000.0, 0.02)
        s = fm_modulate(m, 48000.0, carrier_freq=8000.0, freq_deviation=1000.0, amplitude=0.7)
        assert np.max(np.abs(s)) <= 0.7 + 1e-12, "幅度不得超过 A"
        assert np.max(np.abs(s)) == pytest.approx(0.7, rel=1e-3)

    def test_quadrature_amplitude_bounded(self):
        """正交法与直接法必须逐点等价（峰值仅受采样点位置限制，故用 1e-3 相对容差）。"""
        m = make_message(48000.0, 0.02)
        s_dir = fm_modulate(m, 48000.0, carrier_freq=8000.0, freq_deviation=1000.0, amplitude=0.7)
        s_qua = fm_modulate(m, 48000.0, carrier_freq=8000.0, freq_deviation=1000.0,
                            amplitude=0.7, method="quadrature")
        assert np.max(np.abs(s_qua)) <= 0.7 + 1e-12
        assert np.max(np.abs(s_dir - s_qua)) < 1e-12
        assert np.max(np.abs(s_qua)) == pytest.approx(0.7, rel=1e-3)

    def test_phase_offset_applied(self):
        m = make_message(48000.0, 0.01)
        s0 = fm_modulate(m, 48000.0, carrier_freq=8000.0, freq_deviation=1000.0)
        p0, _, _ = fm_phase(m, 48000.0, 8000.0, 1000.0)
        assert np.allclose(s0, np.cos(p0))
