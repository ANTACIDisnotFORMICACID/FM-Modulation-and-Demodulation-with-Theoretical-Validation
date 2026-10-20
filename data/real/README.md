# 外部广播 IQ 录音

本地文件 `fm_rds_250k_1Msamples.iq` 来自 [PySDR 教程引用的公开仓库](https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq)。[教程](https://www.pysdr.org/content/rds)说明，该录音对应 99.5 MHz 电台，IQ 采样率为 250 kHz，包含 1,000,000 个 `complex64` 采样点。文件大小为 8,000,000 字节，对应 4 秒。原仓库没有明确声明该录音的再分发许可，因此原始 IQ 只保存在本地，不纳入 Git。

在项目根目录使用 Windows PowerShell 下载：

```powershell
New-Item -ItemType Directory -Force data/real | Out-Null
Invoke-WebRequest 'https://raw.githubusercontent.com/777arc/498x/master/fm_rds_250k_1Msamples.iq' -OutFile data/real/fm_rds_250k_1Msamples.iq
```

本项目使用的文件 SHA-256 为 `c336b7be8e08081521465916921bf995253701eb84d218cf73a02dde958a890b`。该哈希用于确认外部验证样本的版本；若上游文件发生变化，应重新检查格式和测量结果。运行接收机：

```powershell
python -m fm_project receive --input data/real/fm_rds_250k_1Msamples.iq --iq-rate 250000 --format cf32le --center-freq 99500000 --decimation 5 --audio-gain 0.5 --output results/sdr/real_fm_recovered.wav --report results/sdr/real_fm_report.json --figure results/figures/real_fm_receiver.png
```

这是一份第三方真实射频录音；`data/generated/` 下的文件则是本项目的软件仿真输出。上述命令运行 `fm_project.receiver.BroadcastFMReceiver`，并未复制外部教程恢复出的音频。
