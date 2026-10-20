# 测试

在项目根目录运行 `python -m pytest tests -q`。共 109 项测试，覆盖原有实通带实验和复基带软件收发链路。

| 文件 | 测试项数 | 范围 | 主要检查内容 |
|---|---:|---|---|
| `test_modulation.py` | 16 | 调制 | 各分量瞬时频率幅度比为 1.0000；积分归一化错误的回归测试 |
| `test_demodulation.py` | 15 | 解调 | 往返 NRMSE < 0.02；最佳拟合比例可恢复 Δf；比较四种解调方法；复数 IQ 的 Q 分量未丢失 |
| `test_channel_spectra.py` | 32 | 信道与频谱 | AWGN 标定往返误差 < 0.1 dB；两种 FFT 归一化方式；帕塞瓦尔定理；贝塞尔边带吻合程度 |
| `test_metrics.py` | 35 | 指标 | NRMSE、对齐、频率偏移、SNR 及指标错误的回归测试 |
| `test_sdr.py` | 11 | 复基带软件收发 | 分块状态、重采样、IQ 文件与模拟接口、广播接收及命令行 |
| **合计** | **109** | | |

依赖见根目录的 [`requirements.txt`](../requirements.txt)。核心测试不需要可选的声卡库 `sounddevice`。
