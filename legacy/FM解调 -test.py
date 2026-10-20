import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import hilbert


# FM调制函数
def fm_modulate(stream, A, f0, f1, fs, T):
    """
    FM调制
    :param stream: 输入信号，通常是连续的正弦波
    :param A: 信号振幅
    :param f0: 中间频率
    :param f1: 频率偏移
    :param fs: 采样频率
    :param T: 信号持续时间
    :return: 调制后的信号
    """
    t = np.arange(0, T, 1/fs)
    integral_of_input = np.cumsum(stream) / fs  # 计算原信号的积分
    fm_signal = A * np.cos(2 * np.pi * f0 * t + 2 * np.pi * f1 * integral_of_input)  # 生成FM信号
    
    return fm_signal

# FM解调函数(希尔伯特)
def fm_demodulate(fm_modulated_signal, fs, fc):
    """
    使用希尔伯特变换和相位差分进行FM解调
    :param fm_modulated_signal: 输入的FM调制信号
    :param fs: 采样频率
    :param fc: 载波频率
    :return: 解调后的信号
    """
    # 使用希尔伯特变换获取信号的瞬时相位
    analytic_signal = hilbert(fm_modulated_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # 通过相位差分计算瞬时频率
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2 * np.pi)
    
    # 去掉一个采样点，使时间匹配
    demodulated_signal = instantaneous_frequency - fc
    
    return demodulated_signal


# 参数
fs = 200  # 采样频率
T = 2  # 信号时长
t = np.arange(0, T, 1/fs)  # 时间序列
f0 = 10  # 载波频率
f1 = 5  # 频率偏移

stream = np.sin(2 * np.pi * 1 * t)  # 输入信号(sin,F=1Hz)

modulated_signal = fm_modulate(stream, 1, f0, f1, fs, T)

demodulated_signal = fm_demodulate(modulated_signal, fs, f0)


#绘制时域图像
plt.figure(figsize=(12, 6))
# 原始信号
plt.subplot(3, 1, 1)
plt.plot(t, stream)
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 调制信号
plt.subplot(3, 1, 2)
plt.plot(t, modulated_signal)
plt.title("FM Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 解调后的信号
plt.subplot(3, 1, 3)
plt.plot(t[:-1], demodulated_signal)
plt.title("FM Demodulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
