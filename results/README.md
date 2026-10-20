# 实验结果

本目录保存 `scripts/` 中各实验脚本生成的文件。需要更新结果时，应重新运行脚本，不要直接修改输出文件。运行 `python scripts/run_all.py` 可重建原有的 `exp01`–`exp05` 实验；运行 `python scripts/run_sdr_experiments.py` 可重建复基带实验 E1–E7。

- `figures/` 保存 `exp01`–`exp05` 的 PNG 和 PDF 图像，以及真实 IQ 接收的 PNG 诊断图。
- `data/` 保存按实验编号命名的 CSV 和 JSON 数值结果。
- `sdr/` 保存 E1–E7 的 CSV、配置文件、PNG 曲线，以及软件收发和真实 IQ 接收的报告。派生 WAV 只保存在本地。复现真实 IQ 接收结果需要 [`data/real/README.md`](../data/real/README.md) 所述录音。

各实验脚本定义了参数、采样率和随机种子；复基带实验还会保存 `sdr/experiment_config.json`。
