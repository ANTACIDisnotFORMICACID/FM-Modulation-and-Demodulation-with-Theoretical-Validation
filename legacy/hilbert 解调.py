
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

def fm_modulate(stream, A, f0, f1, fs, T):
    t = np.arange(0, T, 1/fs)
    integral_of_input = np.cumsum(stream) / fs
    fm_signal = A * np.cos(2 * np.pi * f0 * t + 2 * np.pi * f1 * integral_of_input)
    return fm_signal

def add_noise(signal_data, noise_level):
    N = len(signal_data)
    noise_I = np.random.normal(0, 1, N)
    noise = noise_I
    return signal_data + noise * np.sqrt(noise_level)

def hilbert_fm_demodulate(fm_modulated_signal, fs, fc):
    """
    通过希尔伯特变换解调FM信号

    :param fm_modulated_signal: 调制后的信号，带有复数噪声
    :param fs: 采样频率
    :param fc: 载波频率
    :return: 解调后的基带信号
    """
    # 使用希尔伯特变换获得信号的解析信号
    analytic_signal = hilbert(fm_modulated_signal)
    
    # 计算瞬时相位
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # 计算瞬时频率
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2 * np.pi)  # 获取瞬时频率
    
    # 减去载波频率
    demodulated_signal = instantaneous_frequency - fc  # 减去载波频率，得到基带信号
    
    return demodulated_signal

def low_pass_filter(signal_data, fs, cutoff_freq, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal_data)


# 测试代码：
fs = 300  # 采样频率
T = 2  # 信号时长
t = np.arange(0, T, 1/fs)  # 时间序列
f0 = 20  # 载波频率
f1 = 5  # 频率偏移

# 生成输入信号 (一个简单的正弦波)
stream = np.sin(2 * np.pi * 1 * t)  # 输入信号 (sin, F=1Hz)

# 调制FM信号
modulated_signal = fm_modulate(stream, 1, f0, f1, fs, T)

# 向信号添加噪声
noisy_signal = add_noise(modulated_signal, noise_level=0.05)

# 使用希尔伯特变换解调信号
demodulated_signal = hilbert_fm_demodulate(noisy_signal, fs, f0)

# 设置滤波器的截止频率（滤波器的作用是去除高频噪声）
cutoff_freq = 30  # 截止频率，选择一个合适的值
#filtered_signal = low_pass_filter(demodulated_signal, fs, cutoff_freq)
filtered_signal = demodulated_signal
# 绘制时域图像
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# 原始信号
plt.subplot(5, 1, 1)
plt.plot(t, stream)
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 调制信号
plt.subplot(5, 1, 2)
plt.plot(t, modulated_signal)
plt.title("FM Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 添加复数噪声后的信号
plt.subplot(5, 1, 3)
plt.plot(t, np.real(noisy_signal))  # 只绘制实部
plt.title("FM Modulated Signal with Complex Noise")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 解调后的信号（未滤波）
plt.subplot(5, 1, 4)
plt.plot(t[:-1], demodulated_signal)
plt.title("FM Demodulated Signal (Before Filter)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 滤波后的解调信号
plt.subplot(5, 1, 5)
plt.plot(t[:-1], filtered_signal)
plt.title("FM Demodulated Signal (After Filter)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 创建FM调制信号
fs = 300  # 采样频率
T = 2  # 信号时长
t = np.arange(0, T, 1/fs)
f0 = 20  # 载波频率
f1 = 5  # 频率偏移

# 输入信号 (一个简单的正弦波)
stream = np.sin(2 * np.pi * 1 * t)  # 输入信号 (sin, F=1Hz)

# 调制FM信号
integral_of_input = np.cumsum(stream) / fs
fm_signal = np.cos(2 * np.pi * f0 * t + 2 * np.pi * f1 * integral_of_input)

# 使用Hilbert变换生成解析信号
analytic_signal = hilbert(fm_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))

# 通过相位差分计算瞬时频率
instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2 * np.pi)

# 绘图展示结果
plt.figure(figsize=(10, 8))

# 原始信号
plt.subplot(4, 1, 1)
plt.plot(t, stream)
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 调制后的FM信号
plt.subplot(4, 1, 2)
plt.plot(t, fm_signal)
plt.title("FM Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# 解调后的瞬时频率
plt.subplot(4, 1, 3)
plt.plot(t[1:], instantaneous_frequency)  # 使用差分后的频率
plt.title("Instantaneous Frequency (Demodulated)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.grid(True)

# 绘制解调后的信号（瞬时频率减去载波频率）
demodulated_signal = instantaneous_frequency - f0
plt.subplot(4, 1, 4)
plt.plot(t[1:], demodulated_signal)
plt.title("FM Demodulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
"""
