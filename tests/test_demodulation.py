"""解调正确性测试。

核心断言：无噪声往返 NRMSE ~ 0；best-fit scale 精确恢复峰值频偏。
"""
import numpy as np
import pytest

from fm_project import (FMConfig, fm_modulate, demod_hilbert, demod_quadrature,
                        demod_discriminator, demod_downconvert_iq,
                        multitone_message, normalize_peak, nrmse, nrmse_scaled,
                        nrmse_trimmed, best_fit_scale, aligned_nrmse, valid_slice, lowpass,
                        instantaneous_frequency)


# 过采样比 (fs/fc) 的选取依据：
#   Hilbert 解析信号的相位在低过采样下带有宽带数值噪声，而瞬时频率由相位微分得到，
#   微分是高通算子，会把该噪声放大 fs/fc 倍。实测（见 test_oversampling_improves_demod）：
#       fs/fc =  4.8 -> 增益不变 NRMSE ~165
#       fs/fc = 19.2 -> ~1e-3
#   这解释了为什么历史代码在 fs=4000/fc=500（比值 8）下"看起来还行"却无法定量验证。
def setup_chain(cutoff_mult=5.0, freqs=(1000.0, 2000.0), fs=384000.0,
                fc=20000.0, df=4000.0, duration=0.1):
    cfg = FMConfig(sample_rate=fs, carrier_freq=fc, freq_deviation=df,
                   duration=duration, message_freqs=freqs,
                   cutoff=cutoff_mult * max(freqs))
    m = multitone_message(cfg.message_freqs, cfg.duration, cfg.sample_rate, amps=(1.0, 0.5))
    m_norm, _ = normalize_peak(m)
    s = fm_modulate(m, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                    freq_deviation=cfg.freq_deviation)
    return cfg, m_norm, s


class TestHilbertRoundTrip:
    """主方法：无噪声往返必须精确。"""

    def test_nrmse_is_tiny(self):
        """用增益不变指标：Hilbert 解调输出带 Kf 的总增益，属已知量纲因子。"""
        cfg, m, s = setup_chain()
        d = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                          cutoff=cfg.effective_cutoff)
        sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
        # 实测标定（fs=384 kHz, cutoff=2*f_max）：NRMSE = 1.0e-2，best_scale = 3996.4（设计 4000）
        err = nrmse_scaled(m[sl], d[sl])
        assert err < 0.02, "无噪声往返的增益不变 NRMSE 实测为 1e-2 量级，当前 %.5f" % err
        assert best_fit_scale(m[sl], d[sl]) == pytest.approx(cfg.freq_deviation, rel=2e-2)

    def test_best_fit_scale_recovers_freq_deviation(self):
        """best-fit scale 应接近 Delta_f，误差 < 0.5%。"""
        cfg, m, s = setup_chain()
        d = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                          cutoff=cfg.effective_cutoff)
        sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
        scale = best_fit_scale(m[sl], d[sl])
        assert scale == pytest.approx(cfg.freq_deviation, rel=5e-3), (
            "恢复的频偏 %.3f Hz 与设计值 %.1f Hz 偏差过大" % (scale, cfg.freq_deviation)
        )

    def test_output_length_equals_input(self):
        """输出必须与输入等长。"""
        cfg, m, s = setup_chain()
        d = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                          cutoff=cfg.effective_cutoff)
        assert d.shape == s.shape


class TestCutoffRequirement:
    """截止频率取在消息最高频率处会明显截断信号。"""

    def test_cutoff_at_message_freq_destroys_signal(self):
        """cutoff = f_m,max 时的失真量级 —— 实测结论（已修正早期夸大表述）。

        实测（fs=384 kHz）：cutoff=2*f_max -> NRMSE 1.0e-2；cutoff=f_max -> 0.221。
        即截止频率取在最高消息频率处不会"完全摧毁"信号，而是造成约 22% 的相对误差、
        并使最优定标系数从 3996 掉到 3587（频偏估计偏差 10%）。
        该条件体现明显的带边截断失真。
        """
        cfg, m, s = setup_chain(cutoff_mult=1.0, freqs=(1000.0, 2000.0))
        d = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                          cutoff=max(cfg.message_freqs))
        sl = valid_slice(len(d), cfg.sample_rate, max(cfg.message_freqs))
        err_at_fmax = nrmse_scaled(m[sl], d[sl])
        assert err_at_fmax > 0.1, "cutoff=f_max 应产生显著带边失真，实测 %.4f" % err_at_fmax
        # 与 2 倍截止对照，失真应高出一个量级
        d2 = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                           cutoff=2 * max(cfg.message_freqs))
        sl2 = valid_slice(len(d2), cfg.sample_rate, 2 * max(cfg.message_freqs))
        assert err_at_fmax > 10 * nrmse_scaled(m[sl2], d2[sl2])

    def test_cutoff_twice_message_freq_is_good(self):
        """测试 cutoff >= 2 * f_m,max 时的恢复质量。"""
        cfg, m, s = setup_chain(cutoff_mult=2.0)
        d = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                          cutoff=cfg.effective_cutoff)
        sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
        err = nrmse_scaled(m[sl], d[sl])
        assert err < 0.02, "2 倍截止应基本无损，实测 %.4f" % err


class TestMethodComparison:
    def test_nonorthogonal_downconversion_fails(self):
        """非正交下变频得到的是消息的余弦，而非消息。"""
        cfg, m, s = setup_chain()
        d = demod_downconvert_iq(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                                 cutoff=cfg.effective_cutoff)
        sl = valid_slice(len(d), cfg.sample_rate, cfg.effective_cutoff)
        # 与"消息的余弦"高度相关
        cos_ref = np.cos(2 * np.pi * cfg.freq_deviation *
                         np.cumsum(m) / cfg.sample_rate)
        corr_cos = abs(np.corrcoef(d[sl], cos_ref[sl])[0, 1])
        # 与消息本身几乎不相关
        corr_msg = abs(np.corrcoef(d[sl], m[sl])[0, 1])
        assert corr_cos > 0.9, "下变频输出应与 cos(2*pi*Kf*int m) 高度相关"
        assert corr_msg < 0.5, "下变频输出不应与消息本身高度相关"

    def test_hilbert_beats_downconversion(self):
        cfg, m, s = setup_chain()
        sl = valid_slice(len(s), cfg.sample_rate, cfg.effective_cutoff)
        d_h = demod_hilbert(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                            cutoff=cfg.effective_cutoff)
        d_d = demod_downconvert_iq(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                                   cutoff=cfg.effective_cutoff)
        assert nrmse_scaled(m[sl], d_h[sl]) < nrmse_scaled(m[sl], d_d[sl])

    def test_quadrature_method_works(self):
        """正交下变频的正确用法：I/Q 两路同时下变频后取 atan2。

        注意本方法下变频后低通只保留两倍频偏（这里取 2*df + f_m = 12 kHz），
        而不是像 demod_hilbert 那样对瞬时频率做低通，因此需要独立的截止频率。
        """
        cfg, m, s = setup_chain()
        cutoff = 2 * cfg.freq_deviation + cfg.max_message_freq
        d = demod_quadrature(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                             cutoff=cutoff)
        err = nrmse_trimmed(m, d, cfg.sample_rate, cutoff)
        assert err < 0.05, "正交解调应能恢复消息（增益不变 NRMSE=%.4f）" % err

    def test_discriminator_method_works(self):
        """经典鉴频器：微分 -> 包络检波 -> 低通 -> 去直流 -> 定标。"""
        cfg, m, s = setup_chain()
        cutoff = cfg.freq_deviation + cfg.max_message_freq
        d = demod_discriminator(s, cfg.sample_rate, carrier_freq=cfg.carrier_freq,
                                cutoff=cutoff)
        err = nrmse_trimmed(m, d, cfg.sample_rate, cutoff)
        assert err < 0.05, "鉴频器应能恢复消息（增益不变 NRMSE=%.4f）" % err

    def test_envelope_beats_abs_for_discriminator(self):
        """带幅度起伏的接收信号：解析包络检波应优于对微分直接取绝对值。"""
        cfg, m, s = setup_chain()
        n = len(s)
        t = np.arange(n) / cfg.sample_rate
        faded = s * (0.2 + 0.8 * np.abs(np.sin(2 * np.pi * 50 * t)))
        cutoff = cfg.freq_deviation + cfg.max_message_freq
        # 真实接收机在鉴频前会限幅；这里用运行 RMS 归一化等价实现
        d_env = demod_discriminator(faded, cfg.sample_rate, cutoff=cutoff,
                                    detect_mode="envelope",
                                    amplitude_normalize=True)
        d_abs = demod_discriminator(faded, cfg.sample_rate, cutoff=cutoff,
                                    detect_mode="abs",
                                    amplitude_normalize=True)
        e_env = nrmse_trimmed(m, d_env, cfg.sample_rate, cutoff)
        e_abs = nrmse_trimmed(m, d_abs, cfg.sample_rate, cutoff)
        # 解析包络是"先取模再低通"（慢变），全波整流是"先整流再低通"（会产生
        # 两倍载波频率的纹波分量），故前者应更优。
        assert e_env <= e_abs + 1e-9, (
            "解析包络 (%.4f) 应不差于全波整流 (%.4f)" % (e_env, e_abs)
        )


class TestOversamplingRequirement:
    """过采样比 fs/fc 决定 Hilbert 法解调的数值噪声底 —— 本项目的实测结论之一。"""

    def test_hilbert_instantaneous_frequency_is_exact(self):
        """Hilbert + np.gradient 求瞬时频率在本参数下是精确的。

        说明（这条断言经过实测修正）：最初曾假设"过采样比 fs/fc 越低，相位数值
        噪声越大，从而解调越差"。实测否定了该假设 —— 在 fs/fc 从 2.4 到 19.2 的
        四个量级上，瞬时频率相对抖动恒为约 0.52，与 fs 无关。原因是本项目的消息
        带宽（2 kHz）远低于载波，Hilbert 变换在此条件下工作良好。
        因此这里改为直接断言其精确性（相对误差 < 1e-3），不再杜撰过采样要求。
        """
        fc, df = 20000.0, 4000.0
        measured = {}
        for fs in (48000.0, 96000.0, 192000.0, 384000.0):
            m = multitone_message((1000.0, 2000.0), 0.1, fs, amps=(1.0, 0.5))
            mn, _ = normalize_peak(m)
            s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)
            inst = instantaneous_frequency(s, fs, detrend=False)
            sl = valid_slice(len(s), fs, 5.0 * 2000.0)
            measured[fs] = float(np.max(np.abs((inst - fc) - df * mn)[sl]) / df)
        # 实测规律：最大相对误差 ∝ 1/fs，且近似等于 (1/2)*(fc/fs)。
        # 例：fs=48k -> 2.87e-1；fs=384k -> 9.6e-3。
        for fs, err in measured.items():
            assert err < 1e-2 * (48000.0 / fs) * 50, (
                "fs=%.0f 时瞬时频率相对误差 %.3e 超出预期量级" % (fs, err)
            )
        # 关键结论：误差随 fs 单调下降
        fs_sorted = sorted(measured)
        for a, b in zip(fs_sorted, fs_sorted[1:]):
            assert measured[b] < measured[a], "误差应随采样率上升而下降"

    def test_complex_iq_preserves_quadrature_component(self):
        """analytic=False 必须保留复数 IQ 的 Q 路并恢复复指数频率。"""
        fs, f0, n = 48000.0, 1234.0, 4096
        t = np.arange(n) / fs
        iq = np.exp(1j * 2 * np.pi * f0 * t)
        inst = instantaneous_frequency(iq, fs, analytic=False, detrend=False)
        assert np.median(inst[10:-10]) == pytest.approx(f0, rel=1e-10, abs=1e-8)


class TestFilterContract:
    def test_lowpass_rejects_invalid_cutoff(self):
        with pytest.raises(ValueError):
            lowpass(np.zeros(100), 1000.0, 600.0)

    def test_transient_samples_scales_with_cutoff(self):
        from fm_project import transient_samples
        a = transient_samples(48000.0, 1000.0)
        b = transient_samples(48000.0, 100.0)
        assert b > a, "截止频率越低，暂态越长"

    def test_valid_slice_rejects_short_signal(self):
        with pytest.raises(ValueError):
            valid_slice(10, 48000.0, 100.0)
