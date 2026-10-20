# FM 调制解调及理论验证

本项目用 Python 实现 FM 调制解调实验和离线复基带 FM 软件收发链路，涵盖连续相位 IQ 生成、可配置的信道扰动、解调、WAV 输出与可重复的测量。项目也能处理第三方广播 FM 录制的 IQ 数据，无需本地 SDR 硬件。

本项目源自本人2024年大一期间初步接触业余无线电和信息学时在社团做的FM调制解调算法，故原始的legacy代码较为混乱、版本管理不清，且并未进行实际的无线电通信验证。现对通信有了更深的认识，且取得了业余无线电操作证书A证，于是决定对代码进行重新整理和软件上的通信测试，后续预计使用实际 SDR 或 UVK6 + AIOC 声卡进行实际的无线电通信验证。

## 项目内容

- **软件收发链路：** 单声道 WAV 或生成的测试音 → 抑制镜像的插值 → 复数 FM IQ → 模拟信道 → 相位差鉴频 → 抗混叠抽取 → WAV。
- **离线广播接收：** 已录制 IQ → 信道选择 → FM 鉴频 → 音频高通与去加重 → 单声道 WAV。
- **信道与测量工具：** 复数 AWGN、载波频偏、初始相位、幅度缩放、带宽限制、IQ 与音频 SNR、NMSE、PSD，以及处理耗时比较。
- **理论实验：** 实通带 FM、四种解调方法、贝塞尔边带、卡森带宽和噪声门限研究。

软件链路与实验已在本地测试。项目使用 [PySDR 引用的真实 IQ 录音](https://www.pysdr.org/content/rds/) 生成了可播放、非静音的 4 秒 WAV；尚未通过人工听音评估节目内容，也未测试本地射频发射机或接收机。

## 快速开始

已测试的环境为 Python 3.13。可在虚拟环境或现有 Python 环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

下文为简洁起见使用 `python`；如使用虚拟环境，请换成对应的解释器路径。

### 生成并接收复数 FM 信号

```powershell
python -m fm_project simulate --output results/sdr/demo_recovered.wav --iq-out data/generated/demo.iq --report results/sdr/demo_report.json
```

未指定输入 WAV 时，`simulate` 会生成 1 kHz 与 2 kHz 组成的测试信号。处理已有的单声道或立体声 WAV：

```powershell
python -m fm_project simulate --input input.wav --output recovered.wav --iq-out sent.iq --snr-db 15
```

立体声输入会混合为单声道。IQ 以小端 `complex64` 保存，并配有同名 `.iq.json` 元数据文件。运行 `--help` 可查看载波频偏、带宽、调制频偏、分块长度等参数。因果重采样会引入固定延迟；逐样本比较 WAV 前，应先对齐两段音频。

### 解调真实广播录音

[样本来源与 SHA-256 说明](data/real/README.md)记录了一份与 99.5 MHz 电台相关的 IQ 录音：采样率 250 kHz，格式为 `complex64`，共 1,000,000 点。录音需单独下载；原始 IQ 和恢复的音频不纳入 Git。

```powershell
New-Item -ItemType Directory -Force data/real | Out-Null
Invoke-WebRequest 'https://raw.githubusercontent.com/777arc/498x/master/fm_rds_250k_1Msamples.iq' -OutFile data/real/fm_rds_250k_1Msamples.iq
python -m fm_project receive --input data/real/fm_rds_250k_1Msamples.iq --iq-rate 250000 --format cf32le --center-freq 99500000 --output results/sdr/real_fm_recovered.wav --audio-gain 0.5 --report results/sdr/real_fm_report.json --figure results/figures/real_fm_receiver.png
```

该命令由 4 秒 IQ 录音生成采样率为 50 kHz 的单声道音频。仓库中的[接收报告](results/sdr/real_fm_report.json)记录处理参数和信号电平；[诊断图](results/figures/real_fm_receiver.png)展示 IQ 频谱、鉴频器输出和恢复的波形。明确指定格式和采样率后，`receive` 还支持交织的有符号 8 位（`cs8`）和小端有符号 16 位（`ci16le`）IQ。

## 复现实验

```powershell
python scripts/run_all.py
python scripts/run_sdr_experiments.py
python -m pytest tests -q
```

`run_all.py` 重新生成原有的实通带实验（`exp01`–`exp05`）。`run_sdr_experiments.py` 重新生成复基带实验 E1–E7；CSV、配置文件和图像保存在 [results/sdr/](results/sdr/)。脚本中固定了随机种子。当前测试集包含 **109 项通过的测试**。

| 实验 | 已记录结果 | 结果解释 |
|---|---|---|
| E1：无噪声恢复 | IQ 采样率为 120/240/480 kHz 时，消息信号最大误差 ≤ 9.33×10⁻¹⁵ | 在这些条件下，相位差解调的误差达到浮点精度量级。 |
| E2：复数 IQ 的 AWGN | 输入 IQ SNR 从 0 dB 增至 30 dB 时，平均输出音频 SNR 从 4.87 dB 增至 52.07 dB | 每个点重复 3 次；详见 [CSV](results/sdr/e2_snr.csv)。 |
| E3：载波频偏 | 频偏为 20 kHz 时，未校正 NMSE 为 0.719；校正**已知频偏**后，误差降至数值零 | 尚未实现盲频偏估计。 |
| E4：FM 调制频偏 | IQ SNR 为 10 dB 时，调制频偏从 5 kHz 增至 95 kHz，输出 SNR 从 8.27 dB 增至 33.84 dB | 这一趋势仅对应测试所用的采样率和消息带宽。 |
| E5：鉴频方法比较 | 乘积法与展开相位差法的输出 SNR 均为 37.10 dB；较早的复梯度法为 34.38 dB | 当相位增量不发生回绕时，前两种方法在数学上等价。 |
| E6：信道带宽 | 扫描 30–210 kHz，输出 SNR 范围为 12.26–42.50 dB | 测量曲线并非处处单调；详见 [CSV](results/sdr/e6_bandwidth.csv)。 |
| E7：低 SNR 扫描 | 输入 IQ SNR 从 4 dB 增至 8 dB 时，输出 SNR 从 14.06 dB 变为 28.80 dB | 这组仿真呈现类似门限的变化，不能据此认为门限值具有普遍性。 |

E1–E7 使用时长 0.20 秒、由 700 Hz、1700 Hz 和扫频信号组成的消息；通常采用 240 kHz IQ 采样率与 48 kHz 音频采样率。音频 SNR 在去除固定长度的瞬态后，将含噪接收输出与同一接收机的无噪输出比较，不拟合增益系数。精确实验条件见[配置文件](results/sdr/experiment_config.json)和各 CSV。早期实验的数据与图像分别位于 [results/data/](results/data/) 和 [results/figures/](results/figures/)。

## 信号模型与目录结构

复基带调制器在不同数据块之间保持连续的相位状态：

```text
phase[n] = phase[n-1] + 2π · deviation · message[n] / iq_rate
iq[n]    = exp(j · phase[n])
```

接收机通过 `angle(iq[n] · conj(iq[n-1]))` 估计每个采样点的相位增量。输入消息采用固定幅度范围（`|message| ≤ 1`）；若逐块按峰值归一化，有效频偏量会随之变化。相位增量必须保持在无歧义区间内，以免发生混叠。

| 路径 | 用途 |
|---|---|
| [`fm_project/`](fm_project/) | 可复用的 DSP、信道、接收机、IQ 读写和命令行程序。DSP 类在数据块之间保留相位与滤波器状态。 |
| [`scripts/`](scripts/) | 指定参数、运行扫描并生成结果文件的实验脚本。 |
| [`tests/`](tests/) | 数值计算和接口的回归测试。 |
| [`results/`](results/) | 纳入仓库的图像、CSV 与简要报告。 |
| [`legacy/`](legacy/) | 保留的早期探索脚本；文件演变见[说明](legacy/README.md)。 |

本项目尚未实现立体声广播解码、RDS 解码、实时 SDR 驱动或射频发射。附带的声卡工具是独立的早期实验，使用时需要安装 `sounddevice`；其结果不能用来验证射频链路。
