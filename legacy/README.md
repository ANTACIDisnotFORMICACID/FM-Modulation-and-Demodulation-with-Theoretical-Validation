# FM 实验文件的演变

本目录来自于本人刚接触信息学时的尝试，代码管理相当混乱，留此作为纪念。

保存了 `fm_project` 软件包建立之前使用的独立脚本。文件名和格式保留原样，便于对照早期实验与后来模块化实现的关系。

| 早期文件 | 当时的实验内容 | 现有对应位置 |
|---|---|---|
| `FM 调制.py`、`Example 2.py`、`FM test 6.py`、`FM test4.py`、`FM test5*.py` | 实通带 FM 波形、消息积分及不同参数的重复实验 | `fm_project/modulation.py`、`fm_project/config.py` |
| `FM解调 -test.py`、`hilbert 解调.py`、`FM 解调-test2.py`、`FM test3.py`、`fm.py` | 希尔伯特相位法、单路下变频法与导数包络法的解调尝试 | `fm_project/demodulation.py` |
| `how to use fft.py`、`Power Spectral Density.py`、`FM 调制-PSD-tip.pdf` | 频谱与功率谱密度实验 | `fm_project/spectra.py`、`scripts/exp04_spectrum_bessel.py` |
| `IQ save and read test.py`、`IQ save and read with tofile &fromfile.py`、`original_signal IQ.py`、`original.iq`、`output.iq` | 交织整数采样和原始复数采样的读写实验 | `fm_project/iqio.py` |
| `whole test.py` | 使用另一组采样率和频率参数的独立全流程实验 | `scripts/exp01_basic_chain.py` 及后续软件收发链路 |
| `FSK 1.py` | 独立的 BFSK 实验 | 仅保留在本目录 |

早期 FM 脚本曾复制并修改相近的调制、解调函数；软件包建立后，这些实现由具名算法及明确的参数、测量模块取代。本目录中的 IQ 文件属于早期实验；项目使用的外部广播 IQ 录音另见 `data/real/`。
