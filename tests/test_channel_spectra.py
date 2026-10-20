"""信道与频谱测试 —— 验证 AWGN 定标与 Bessel 理论谱。"""
import numpy as np
import pytest
from scipy.special import jv

from fm_project import (awgn, measured_snr_db, signal_power, noise_power_for_snr,
                        fft_spectrum, psd_welch, fm_bessel_signal, fm_theoretical_psd,
                        fm_modulate, tone_message, FMConfig, valid_slice)


class TestAWGNCalibration:
    """awgn() 的 SNR 定标必须准确：实测往返误差 < 0.1 dB。"""

    @pytest.mark.parametrize("snr", [-10.0, 0.0, 10.0, 30.0, 50.0])
    def test_real_signal_snr_roundtrip(self, snr):
        rng = np.random.default_rng(1234)
        x = np.sin(2 * np.pi * 1000 * np.arange(200000) / 48000.0)
        y = awgn(x, snr, rng)
        assert measured_snr_db(x, y) == pytest.approx(snr, abs=0.1)

    @pytest.mark.parametrize("snr", [-5.0, 10.0, 40.0])
    def test_complex_signal_snr_roundtrip(self, snr):
        rng = np.random.default_rng(7)
        x = np.exp(1j * 2 * np.pi * 0.01 * np.arange(100000))
        y = awgn(x, snr, rng)
        assert measured_snr_db(x, y) == pytest.approx(snr, abs=0.1)

    def test_noise_power_matches_request(self):
        rng = np.random.default_rng(0)
        x = np.ones(100000)
        p = noise_power_for_snr(x, 20.0)
        y = awgn(x, 20.0, rng)
        assert signal_power(y - x) == pytest.approx(p, rel=0.05)

    def test_reproducible_with_same_seed(self):
        x = np.ones(1000)
        a = awgn(x, 10.0, np.random.default_rng(42))
        b = awgn(x, 10.0, np.random.default_rng(42))
        assert np.array_equal(a, b)

    def test_zero_power_signal_raises(self):
        with pytest.raises(ValueError):
            awgn(np.zeros(100), 10.0, np.random.default_rng(0))


class TestFFTSpectrum:
    def test_single_tone_amplitude_is_accurate(self):
        """加窗补偿后，单音幅度谱峰值应等于真实幅度。"""
        fs, A = 48000.0, 0.7
        # 取 24 个周期（f0 = 24 Hz）：此时信号周期数远小于窗长，Hann 窗主瓣
        # （约 3 条谱线宽）内的能量几乎不丢失，峰值可较准确读出 A。
        n = 48000
        f0 = fs * 24 / n
        x = A * np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        # 测量单条谱线的幅度必须用 normalize="amplitude"；
        # 默认的 "power" 口径会因单边加倍而把峰值读成 A*sqrt(2)
        freq, mag = fft_spectrum(x, fs, window="hann", magnitude="linear",
                                 normalize="amplitude")
        k = np.argmin(np.abs(freq - f0))
        assert mag[k] == pytest.approx(A, rel=0.02)

    def test_power_and_amplitude_conventions_differ_by_sqrt2(self):
        """两种归一化口径的关系必须明确，防止后续混用造成 3 dB 错觉。

        用 boxcar 窗 + 整数周期谱线：此时窗的能量完全集中在一条谱线上，
        sqrt(2) 关系才严格成立。若用 Hann 窗，主瓣跨约 3 条谱线，
        峰值本身会被窗压低（与归一化无关），对比会失去意义。
        """
        fs, n = 48000.0, 48000
        f0 = fs * 24 / n                     # 整数周期 -> 谱线落在频点上
        x = np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        _, mag_amp = fft_spectrum(x, fs, magnitude="linear",
                                  normalize="amplitude", window="boxcar")
        _, mag_pow = fft_spectrum(x, fs, magnitude="linear",
                                  normalize="power", window="boxcar")
        assert np.max(mag_amp) == pytest.approx(1.0, rel=1e-3)
        # 功率口径的峰值 = A/sqrt(2)（正弦的 RMS），这正说明它是 RMS 而非幅度口径
        assert np.max(mag_pow) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-3)
        # 两口径之比恒为 sqrt(2)
        assert np.max(mag_amp) / np.max(mag_pow) == pytest.approx(np.sqrt(2.0), rel=1e-6)
        # 功率口径满足 Parseval：sum(|W_k|^2, 单边) == 信号平均功率 mean(x^2)
        assert np.sum(mag_pow ** 2) == pytest.approx(np.mean(x ** 2), rel=1e-6)

    def test_parseval_holds_for_hann_power_spectrum(self):
        """Hann 窗下（谱线不落在频点上）功率口径仍须满足 Parseval。"""
        fs, n = 48000.0, 48000
        f0 = fs * 24.7 / n
        x = np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        _, mag_pow = fft_spectrum(x, fs, magnitude="linear", normalize="power")
        # 功率口径满足 Parseval：sum(|W_k|^2, 单边) == 信号平均功率 mean(x^2)。
        # 容差 5e-3：非整数周期的正弦其 mean(x^2) 本身偏离 0.5（此处 0.49904），
        # 该差异来自信号而非归一化，故不能取更严的容差。
        assert np.sum(mag_pow ** 2) == pytest.approx(np.mean(x ** 2), rel=5e-3)

    @pytest.mark.parametrize("dc", [0.0, 0.7, 2.0])
    def test_power_convention_parseval_with_dc(self, dc):
        """含直流时 Parseval 仍须成立。

        回归测试：早期实现把单边加倍（sqrt(2)）无差别施加到 DC 谱线上，
        使含直流信号的总功率虚高（实测 0.7+正弦 时 +33%）。
        DC 与 Nyquist 谱线必须单独处理、不得加倍。
        """
        fs, n = 48000.0, 48000
        f0 = fs * 30 / n
        x = dc + np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        _, mag = fft_spectrum(x, fs, magnitude="linear", normalize="power")
        assert np.sum(mag ** 2) == pytest.approx(np.mean(x ** 2), rel=1e-6)

    @pytest.mark.parametrize("dc", [0.0, 1.5])
    def test_amplitude_convention_ignores_dc(self, dc):
        """幅度口径读的是交流谱线峰值，直流不应干扰它。"""
        fs, n = 48000.0, 48000
        f0 = fs * 24 / n
        x = dc + 0.6 * np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        freq, mag = fft_spectrum(x, fs, magnitude="linear",
                                 normalize="amplitude", window="boxcar")
        k = np.argmin(np.abs(freq - f0))
        assert mag[k] == pytest.approx(0.6, rel=1e-3)


class TestDiscriminatorAmplitudeNormalize:
    """回顾审计：amplitude_normalize 一度是"静默无效"的。"""

    def test_normalize_actually_changes_output(self):
        """带衰落的信号上，开启归一化必须改变输出（不能只是常数缩放）。"""
        from fm_project import (FMConfig, tone_message, fm_modulate,
                                demod_discriminator, normalize_peak, valid_slice)
        fs, fc, df, fm = 384000.0, 20000.0, 4000.0, 1000.0
        m = tone_message(fm, 0.05, fs)
        mn, _ = normalize_peak(m)
        s = fm_modulate(mn, fs, carrier_freq=fc, freq_deviation=df)
        t = np.arange(len(s)) / fs
        faded = s * (0.2 + 0.8 * np.abs(np.sin(2 * np.pi * 50 * t)))
        c = df + fm
        d_off = demod_discriminator(faded, fs, cutoff=c, amplitude_normalize=False)
        d_on = demod_discriminator(faded, fs, cutoff=c, amplitude_normalize=True)
        sc = np.dot(d_on - d_on.mean(), d_off - d_off.mean()) / np.var(d_off)
        resid = (d_on - d_on.mean()) - sc * (d_off - d_off.mean())
        rel = np.std(resid) / np.std(d_off)
        assert rel > 0.05, (
            "归一化后输出与未归一化的输出不应只差一个常数因子（实测残差比 %.4f）" % rel
        )

    def test_normalize_improves_faded_demod(self):
        from fm_project import (tone_message, fm_modulate, demod_discriminator,
                                normalize_peak, valid_slice, nrmse_scaled)
        fs, fc, df, fm = 384000.0, 20000.0, 4000.0, 1000.0
        m = tone_message(fm, 0.05, fs)
        mn, _ = normalize_peak(m)
        s = fm_modulate(mn, fs, carrier_freq=fc, freq_deviation=df)
        t = np.arange(len(s)) / fs
        faded = s * (0.2 + 0.8 * np.abs(np.sin(2 * np.pi * 50 * t)))
        c = df + fm
        sl = valid_slice(len(s), fs, c)
        e_off = nrmse_scaled(mn[sl], demod_discriminator(faded, fs, cutoff=c,
                                                         amplitude_normalize=False)[sl])
        e_on = nrmse_scaled(mn[sl], demod_discriminator(faded, fs, cutoff=c,
                                                        amplitude_normalize=True)[sl])
        assert e_on < 0.5 * e_off, (
            "归一化应显著改善衰落下的解调（%.4f -> %.4f）" % (e_off, e_on)
        )

    def test_amplitude_convention_needs_line_on_bin(self):
        """幅度口径只有在谱线落在频点上时才读得准 —— 必须显式记录这一前提。"""
        fs, n = 48000.0, 48000
        on_bin = fs * 24 / n
        off_bin = fs * 24.5 / n
        x_on = np.sin(2 * np.pi * on_bin * np.arange(n) / fs)
        x_off = np.sin(2 * np.pi * off_bin * np.arange(n) / fs)
        _, m_on = fft_spectrum(x_on, fs, magnitude="linear",
                               normalize="amplitude", window="boxcar")
        _, m_off = fft_spectrum(x_off, fs, magnitude="linear",
                                normalize="amplitude", window="boxcar")
        assert np.max(m_on) == pytest.approx(1.0, rel=1e-3)
        assert np.max(m_off) < 0.65, "谱线落在两频点之间时，峰值必然被压低"

    def test_db_axis_peak_is_zero_dbfs_for_unit_amplitude(self):
        fs, n = 48000.0, 48000
        f0 = fs * 24 / n
        x = np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        freq, mag_db = fft_spectrum(x, fs, magnitude="db", normalize="amplitude")
        k = np.argmin(np.abs(freq - f0))
        assert mag_db[k] == pytest.approx(0.0, abs=0.2)

    def test_returns_correct_length_onesided(self):
        fs, n = 48000.0, 1024
        x = np.random.default_rng(0).standard_normal(n)
        freq, _ = fft_spectrum(x, fs)
        assert freq.size == n // 2 + 1


class TestWelchPSD:
    def test_psd_dc_level_of_white_noise(self):
        """白噪声的 PSD 电平应约等于其方差密度。"""
        fs, n = 48000.0, 480000
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        f, p = psd_welch(x, fs, nperseg=4096, db=False)
        # 白噪声双边 PSD = var/fs；单边加倍
        expected = 2.0 * np.var(x) / fs
        assert np.median(p) == pytest.approx(expected, rel=0.1)

    def test_single_tone_peak_location(self):
        fs, n, f0 = 48000.0, 96000, 4000.0
        x = np.sin(2 * np.pi * f0 * np.arange(n) / fs)
        f, p = psd_welch(x, fs, nperseg=8192)
        assert abs(f[np.argmax(p)] - f0) < 2 * fs / 8192


class TestBesselTheory:
    """Bessel 展开与直接调制法必须一致 —— exp04 的理论基准。"""

    def test_bessel_signal_matches_direct_modulation(self):
        fs, fc, df, fm = 480000.0, 20000.0, 4000.0, 1000.0
        cfg = FMConfig(sample_rate=fs, carrier_freq=fc, freq_deviation=df,
                       duration=0.05, message_freqs=(fm,), message_amps=(1.0,))
        m = tone_message(fm, cfg.duration, fs)
        s_dir = fm_modulate(m, fs, carrier_freq=fc, freq_deviation=df)
        s_bes = fm_bessel_signal(df, fm, fs, cfg.duration, carrier_freq=fc)
        sl = slice(100, -100)
        assert np.max(np.abs(s_dir[sl] - s_bes[sl])) < 1e-3

    def test_carrier_component_equals_j0(self):
        """载波分量的幅度应为 |J_0(beta)|。"""
        fs, fc, fm = 960000.0, 50000.0, 1000.0
        for beta in (0.5, 2.0, 5.0):
            df = beta * fm
            n = int(0.2 * fs)
            t = np.arange(n) / fs
            x = fm_bessel_signal(df, fm, fs, 0.2, carrier_freq=fc)
            # 用相干解调提取载波分量的幅度
            amp = 2 * np.abs(np.mean(x * np.exp(-1j * 2 * np.pi * fc * t)))
            assert amp == pytest.approx(abs(jv(0, beta)), rel=0.02)

    def test_sideband_amplitudes_equal_bessel_coefficients(self):
        fs, fc, fm, beta = 960000.0, 50000.0, 1000.0, 3.0
        df = beta * fm
        n = int(0.2 * fs)
        t = np.arange(n) / fs
        x = fm_bessel_signal(df, fm, fs, 0.2, carrier_freq=fc)
        for k in (1, 2, 3, 4, 5):
            for sign in (+1, -1):
                fk = fc + sign * k * fm
                amp = 2 * np.abs(np.mean(x * np.exp(-1j * 2 * np.pi * fk * t)))
                assert amp == pytest.approx(abs(jv(k, beta)), rel=0.03), (
                    "n=%+d 阶边带幅度 %.5f 与 |J_%d(%.1f)|=%.5f 不符"
                    % (sign * k, amp, k, beta, abs(jv(k, beta)))
                )

    def test_theoretical_line_powers_sum_to_one(self):
        """sum_n J_n(beta)^2 = 1，总功率应归一。"""
        _, p = fm_theoretical_psd(4000.0, 1000.0, carrier_freq=20000.0, db=False)
        assert np.sum(p) == pytest.approx(1.0, rel=1e-3)

    def test_theoretical_psd_on_axis_peaks_near_lines(self):
        fc, df, fm = 20000.0, 4000.0, 1000.0
        f = np.linspace(fc - 12000, fc + 12000, 200001)
        _, p = fm_theoretical_psd(df, fm, carrier_freq=fc, freq_axis=f, rbw=80.0, db=False)
        for k in (-4, -1, 0, 1, 4):
            fl = fc + k * fm
            idx = np.argmin(np.abs(f - fl))
            local = p[max(0, idx - 40): idx + 40].max()
            # 该谱线附近应出现局部极大
            assert local > p[idx - 2000] or local > p[idx + 2000]

    def test_carson_bandwidth_contains_most_power(self):
        """Carson 带宽内的功率应占绝大部分（beta=4 时 J_n 衰减很快）。"""
        fc, fm, beta = 20000.0, 1000.0, 4.0
        df = beta * fm
        edge = 2 * (df + fm)
        line_f, line_p = fm_theoretical_psd(df, fm, carrier_freq=fc, db=False)
        inside = np.abs(line_f - fc) <= edge
        assert np.sum(line_p[inside]) > 0.99
