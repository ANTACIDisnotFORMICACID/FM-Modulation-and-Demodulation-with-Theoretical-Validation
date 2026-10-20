# IQ 数据

软件收发链路使用 `fm_project/iqio.py` 保存小端 `complex64` IQ（`cf32le`，每个采样点 8 字节），并在旁边保存同名的 `.iq.json` 元数据文件。读取外部录音时，还支持交织的有符号 8 位（`cs8`）和小端有符号 16 位（`ci16le`）IQ。外部文件需明确指定格式与 IQ 采样率，不能将不同格式混用。

元数据包含 `sample_rate`、`center_freq`、`fmt`、`sample_count` 和 `source`。保存在 `real/` 与 `generated/` 下的原始 IQ 文件由 Git 忽略。外部录音的来源和复现步骤见 [real/README.md](real/README.md)。
