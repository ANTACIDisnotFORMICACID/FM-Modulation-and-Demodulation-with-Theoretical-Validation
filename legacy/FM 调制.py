import numpy as np
import matplotlib.pyplot as plt

# FM调制函数
def fm_modulate(stream, A, f0, f1, fs, T):
    """
    用于FM调制
    :param stream: 输入信号，通常是连续的正弦波
    :param A: 信号振幅
    :param f0: 中间频率
    :param f1: 频率偏移
    :param fs: 采样频率
    :param T: 信号持续时间
    :return: 调制后的信号
    """
    
    # 计算FM调制信号
    integral_of_input = np.cumsum(stream) / fs  # 计算输入信号的积分
    fm_signal = A * np.cos(2 * np.pi * f0 * t + 2 * np.pi * (f1 - f0) * integral_of_input)  # 生成FM信号
    
    return fm_signal  # 返回调制后的信号

# 参数设置
fs = 200  # 采样频率（Hz）
T = 2  # 信号持续时间
t = np.arange(0, T, 1/fs)  # 时间序列
stream = np.sin(2 * np.pi * 1 * t)  # 输入信号：频率为1Hz的正弦波
f0 = 10  # 中间频率（Hz）
f1 = 12  # 频率偏移（Hz）

# 调制
modulated_signal = fm_modulate(stream, 1, f0, f1, fs, T)

# 设置图形宽度和高度
fig_width = 10  # 图形宽度
fig_height = 6  # 图形高度

# 创建第一个子图：原始信号
plt.subplot(2, 1, 1)  # 2行1列的第一个子图
plt.plot(t, stream)
plt.title("Original Signal (Sine Wave)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 创建第二个子图：调制信号
plt.subplot(2, 1, 2)  # 2行1列的第二个子图
plt.plot(t, modulated_signal)
plt.title("FM Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 显示图形
plt.tight_layout()  # 自动调整子图布局
plt.show()

#PSD图像生成

N = len(modulated_signal)
PSD = np.abs(np.fft.fft(modulated_signal))**2 / N
PSD_log = 10.0 * np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.fft.fftfreq(N, 1/fs)#频率平移
f_shifted = np.fft.fftshift(f)

plt.figure(figsize=(16, 6))
plt.plot(f_shifted, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.title("Power Spectral Density (PSD)")
plt.show()
