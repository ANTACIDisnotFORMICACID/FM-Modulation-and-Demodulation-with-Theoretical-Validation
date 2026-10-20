"""指标函数测试 —— 用合成数据锁定每一个指标的定义。

为什么单独测指标：本项目曾两次因指标本身的缺陷而得出错误结论——
  1. nrmse_scaled 的分母用了未定标参考的 std，把 1% 的残差报成 41；
  2. 解调 SNR 的参考功率取了原始消息而非拟合分量，低估约 72 dB。
指标函数如果不被测，它就会成为最隐蔽的错误来源。
"""
import numpy as np
import pytest

from fm_project import (nrmse, nrmse_scaled, nrmse_trimmed, best_fit_scale,
                        aligned_nrmse, estimate_freq_deviation, loopback_snr_db,
                        dbfs, fm_modulate, tone_message, awgn, measured_snr_db,
                        signal_power, noise_power_for_snr, valid_slice)


def _ref(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 1000.0
    return (np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 11 * t)
            + 0.2 * rng.standard_normal(n))


class TestNrmseScaledDegeneracy:
    """估计 = a*ref + c 时，增益不变 NRMSE 必须为 0（机器精度）。"""

    @pytest.mark.parametrize("a", [1.0, -7.0, 4000.0, 1e5])
    @pytest.mark.parametrize("c", [0.0, 100.0, -3.0])
    def test_affine_estimates_are_exact(self, a, c):
        r = _ref()
        assert nrmse_scaled(r, a * r + c) < 1e-10

    @pytest.mark.parametrize("a", [1.0, 4000.0])
    def test_degenerate_with_offset_reference(self, a):
        """参考本身含直流/趋势时也应精确 —— 用于确认去均值处理正确。"""
        t = np.arange(4096) / 1000.0
        for r in (2.0 + np.sin(2 * np.pi * 3 * t),
                  2.0 + 0.5 * t + np.sin(2 * np.pi * 3 * t)):
            assert nrmse_scaled(r, a * r + 7.0) < 1e-10

    def test_gain_invariance_is_the_whole_point(self):
        """同一波形乘以任意增益，指标不得变化。"""
        r = _ref()
        noise = 0.01 * np.random.default_rng(1).standard_normal(r.size)
        base = nrmse_scaled(r, r + noise)
        for a in (10.0, 1e4):
            assert nrmse_scaled(r, a * (r + noise)) == pytest.approx(base, rel=1e-9)

    def test_scale_invariant_metric_known_value(self):
        """对已知噪声电平，读数应约等于 噪声RMS/信号RMS。"""
        rng = np.random.default_rng(2)
        r = np.sin(2 * np.pi * np.arange(8192) / 64)
        w = 0.05
        e = r + w * rng.standard_normal(r.size)
        # 理论值约为 w/std(r)
        assert nrmse_scaled(r, e) == pytest.approx(w / np.std(r), rel=0.08)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            nrmse_scaled(np.ones(10), np.ones(11))

    def test_rejects_zero_variance_reference(self):
        with pytest.raises(ValueError):
            nrmse_scaled(np.ones(100), np.ones(100))


class TestPlainNrmse:
    """标准 NRMSE 要求 estimate 已正确标定。"""

    def test_identical_is_zero(self):
        r = _ref()
        assert nrmse(r, r) == pytest.approx(0.0, abs=1e-12)

    def test_gain_error_shows_up(self):
        r = _ref()
        # 2 倍增益 -> 相对误差约 50% 量级，说明它【不是】增益不变的
        assert nrmse(r, 2.0 * r) > 1.0


class TestBestFitScale:
    def test_recovers_known_gain(self):
        r = _ref()
        for a in (1.0, -3.0, 4000.0):
            assert best_fit_scale(r, a * r) == pytest.approx(a, rel=1e-9)

    def test_recovers_gain_with_offset(self):
        r = _ref()
        assert best_fit_scale(r, 4000.0 * r + 123.0) == pytest.approx(4000.0, rel=1e-9)


class TestAlignedNrmse:
    def test_finds_known_lag(self):
        r = _ref(8192)
        lag_true = 37
        e = np.roll(r, lag_true) + 0.001 * np.random.default_rng(3).standard_normal(r.size)
        err, lag, scale = aligned_nrmse(r, e, max_lag=128)
        assert abs(abs(lag) - lag_true) <= 1
        assert err < 0.05

    def test_rejects_short_signal(self):
        with pytest.raises(ValueError):
            aligned_nrmse(np.ones(20), np.ones(20), max_lag=64)


class TestFreqDeviationEstimation:
    def test_recovers_design_value(self):
        fs, fc, fm = 384000.0, 20000.0, 1000.0
        for beta in (0.5, 2.0, 5.0):
            df = beta * fm
            m = tone_message(fm, 0.05, fs)
            s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)
            est = estimate_freq_deviation(s, fs, fc)
            assert est == pytest.approx(df, rel=2e-3), (
                "beta=%.1f: 估计 %.2f Hz vs 设计 %.2f Hz" % (beta, est, df)
            )

    def test_estimate_is_scale_invariant(self):
        """幅度不应影响频偏估计（FM 是恒包络调制）。"""
        fs, fc, fm = 384000.0, 20000.0, 1000.0
        m = tone_message(fm, 0.03, fs)
        a = estimate_freq_deviation(fm_modulate(m, fs, carrier_freq=fc, freq_deviation=3000.0), fs, fc)
        b = estimate_freq_deviation(0.3 * fm_modulate(m, fs, carrier_freq=fc, freq_deviation=3000.0), fs, fc)
        assert a == pytest.approx(b, rel=1e-2)


class TestSnrUtilities:
    def test_dbfs_of_unit_sine(self):
        """幅度 1 的正弦，RMS 电平应为 -3.01 dBFS。"""
        x = np.sin(2 * np.pi * np.arange(48000) / 16)
        assert dbfs(x) == pytest.approx(-3.0103, abs=0.01)

    def test_dbfs_of_full_scale_square(self):
        assert dbfs(np.ones(1000)) == pytest.approx(0.0, abs=1e-6)

    def test_loopback_snr(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(10000) * 10.0
        noise = rng.standard_normal(10000)
        # 功率比 100 -> 20 dB
        assert loopback_snr_db(sig, noise) == pytest.approx(20.0, abs=0.5)

    def test_noise_power_helper(self):
        x = np.ones(10000)
        assert noise_power_for_snr(x, 10.0) == pytest.approx(0.1, rel=1e-9)


class TestValidSliceContract:
    def test_transient_scales_inversely_with_cutoff(self):
        from fm_project import transient_samples
        assert transient_samples(48000.0, 500.0) > transient_samples(48000.0, 5000.0)

    def test_valid_slice_raises_on_short_signal(self):
        with pytest.raises(ValueError):
            valid_slice(50, 48000.0, 100.0)

    def test_valid_slice_symmetric(self):
        sl = valid_slice(10000, 48000.0, 1000.0)
        assert sl.start == 10000 - sl.stop


class TestMetricAuditRegression:
    """把本次审计中发现并修正的两个指标缺陷固化为回归测试。"""

    def test_hilbert_roundtrip_metric_is_small(self):
        """修正分母后，无噪声往返的读数必须是 1e-2 量级而非 1e2。"""
        fs, fc, fm = 384000.0, 20000.0, 1000.0
        from fm_project import demod_hilbert, normalize_peak
        m = tone_message(fm, 0.05, fs)
        mn, _ = normalize_peak(m)
        s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=4000.0)
        d = demod_hilbert(s, fs, carrier_freq=fc, cutoff=4000.0)
        sl = valid_slice(len(d), fs, 4000.0)
        err = nrmse_scaled(mn[sl], d[sl])
        assert err < 0.05, (
            "若该值变成 1e2 量级，说明 nrmse_scaled 的分母又退回了未定标参考"
        )

    def test_snr_helper_must_include_loop_gain(self):
        """解调 SNR 的参考功率必须含环路增益 s；否则会低估约 20log10(s) dB。"""
        fs, fc, fm = 384000.0, 20000.0, 1000.0
        from fm_project import demod_hilbert, normalize_peak
        m = tone_message(fm, 0.05, fs)
        mn, _ = normalize_peak(m)
        s = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=4000.0)
        r = awgn(s, 30.0, np.random.default_rng(0))
        d = demod_hilbert(r, fs, carrier_freq=fc, cutoff=4000.0)
        sl = valid_slice(len(d), fs, 4000.0)
        ref = mn[sl] - mn[sl].mean()
        est = d[sl] - d[sl].mean()
        sc = float(np.dot(est, ref) / np.dot(ref, ref))
        sig = sc * ref
        noise = est - sig
        snr_out = 10 * np.log10(np.mean(sig ** 2) / np.mean(noise ** 2))
        # 本项目 Hilbert 解调输出单位为 Hz，环路增益应为 Kf 量级（约 4e3）
        assert sc > 1e3, "环路增益异常"
        # 若参考功率误用原始消息，读数会低 20log10(sc) 约 72 dB
        assert snr_out > 0, "修正后高 SNR 输入应给出正的输出 SNR"
