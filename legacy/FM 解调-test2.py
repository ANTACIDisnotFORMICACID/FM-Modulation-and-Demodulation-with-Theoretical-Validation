import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# FM调制函数
def fm_modulate(stream, A, f0, f1, fs, T):
    t = np.arange(0, T, 1/fs)
    integral_of_input = np.cumsum(stream) / fs  # 计算原信号的积分
    fm_signal = A * np.cos(2 * np.pi * f0 * t + 2 * np.pi * f1 * integral_of_input)  # 生成FM信号
    return fm_signal

# FM解调函数 (直接计算复数信号的瞬时相位)
def fm_demodulate(fm_modulated_signal, fs, fc):
    analytic_signal = fm_modulated_signal + 0j  # 确保信号为复数
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # 通过相位差分计算瞬时频率
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2 * np.pi)  # 获取瞬时频率
    demodulated_signal = instantaneous_frequency - fc  # 减去载波频率
    
    return demodulated_signal

# 添加复数噪声
def add_complex_noise(signal_data, noise_level):
    N = len(signal_data)
    noise_I = np.random.normal(0, 1, N)  # 生成实部噪声
    noise_Q = np.random.normal(0, 1, N)  # 生成虚部噪声
    noise = noise_I + 1j * noise_Q  # 合成复数噪声
    return signal_data + noise * np.sqrt(noise_level)

# 低通滤波器设计
def low_pass_filter(signal_data, fs, cutoff_freq, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计低通滤波器
    return filtfilt(b, a, signal_data)  # 应用滤波器

# 参数设置
fs = 300  # 采样频率
T = 2  # 信号时长
t = np.arange(0, T, 1/fs)  # 时间序列
f0 = 20  # 载波频率
f1 = 5  # 频率偏移

# 生成输入信号 (一个简单的正弦波)
stream = np.sin(2 * np.pi * 1 * t)  # 输入信号 (sin, F=1Hz)

# 调制FM信号
modulated_signal = fm_modulate(stream, 1, f0, f1, fs, T)

# 向信号添加复数噪声
noisy_signal = add_complex_noise(modulated_signal, noise_level=0.1)

# 解调信号
demodulated_signal = fm_demodulate(noisy_signal, fs, f0)

# 设置滤波器的截止频率（滤波器的作用是去除高频噪声）
cutoff_freq = 30  # 截止频率，选择一个合适的值
filtered_signal = low_pass_filter(demodulated_signal, fs, cutoff_freq)

# 绘制时域图像
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
