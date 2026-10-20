import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# FM调制函数
def fm_modulate(stream, A, f0, f1, fs, t):
    A_stream = np.max(np.abs(stream))
    integral_of_input = np.cumsum(stream) / (fs * A_stream)  # 计算原信号的积分
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

# 低通滤波器
def low_pass_filter(signal_data, fs, cutoff_freq, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计低通滤波器
    return filtfilt(b, a, signal_data)  # 应用滤波器

# 高通滤波器去除直流分量
def high_pass_filter(signal_data, fs, cutoff_freq, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='high', analog=False)  # 设计高通滤波器
    return filtfilt(b, a, signal_data)  # 应用滤波器

#时域表示图像
def plot_time_domain(signal_data, t, ax, title):
    ax.plot(t,np.real(signal_data))
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

# 频域表示图像（使用FFT）
def plot_frequency_domain(signal_data, fs, ax, title):
    N = len(signal_data)
    freq = np.fft.fftfreq(N, 1/fs)
    signal_freq = np.fft.fft(signal_data)
    signal_freq = np.fft.fftshift(signal_freq)  # 将零频率分量移动到中间
    freq = np.fft.fftshift(freq)  # 同样平移频率轴
    magnitude = np.abs(signal_freq)  # 计算频域幅度
    ax.plot(freq, magnitude)
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    return freq, magnitude  # 返回频率和幅度


# 参数设置
A = 1
fs = 400  # 采样频率
T = 2  # 信号时长
t = np.arange(0, T, 1/fs)  # 时间序列
f0 = 50  # 载波频率
f1 = 5   # 频率偏移

# 生成输入信号 
stream = np.sin(2 * np.pi * 5 * t)  # 输入信号 (sin, F=5Hz)
"""
#参数
A = 1
f0 = 50  # 载波频率
f1 = 5   # 频率偏移

#读取输入信号
output_file = 'original.iq'
fs = 400
stream = np.fromfile(output_file, np.complex64)#注意p ython默认复数格式为complex128以合适格式读取
N = len(stream)
t = np.arange(0,N/fs,1/fs)
"""

# 调制FM信号
modulated_signal = fm_modulate(stream, A, f0, f1, fs, t)

# 向信号添加复数噪声
noisy_signal = add_complex_noise(modulated_signal, noise_level=0.05)

# ******************** 下变频处理 ********************
# 使用同频率的载波进行下变频，获得基带信号
carrier = np.cos(2 * np.pi * f0 * t)  # 载波信号
downconverted_signal = noisy_signal * carrier  # 下变频过程

# ******************** 低通滤波器 ********************
# 应用低通滤波器，去除高频成分
cutoff_freq = 30  # 截止频率
filtered_signal = low_pass_filter(downconverted_signal, fs, cutoff_freq)

# 设置高通滤波器的截止频率
cutoff_freq_hp = 2  # 截止频率,去除直流分量
filtered_signal_hp = high_pass_filter(filtered_signal, fs, cutoff_freq_hp)

from matplotlib import gridspec

# 绘制时域图像与频域图像
fig = plt.figure(figsize=(24, 18))  # 调整画布大小以适应 5x2 的图像阵列

# 使用 gridspec 约束子图大小
gs = gridspec.GridSpec(5, 2, figure=fig)
gs.update(wspace=0.3, hspace=1)

# 原始信号（时域）
ax1 = fig.add_subplot(gs[0, 0])
plot_time_domain(stream, t, ax1, "Original Signal")

# 调制信号（时域）
ax2 = fig.add_subplot(gs[1, 0])
plot_time_domain(modulated_signal, t, ax2, "FM Modulated Signal")

# 添加复数噪声后的信号（时域）
ax3 = fig.add_subplot(gs[2, 0])
plot_time_domain(noisy_signal, t, ax3, "FM Modulated Signal with Complex Noise")

# 滤波后的解调信号（时域）
ax4 = fig.add_subplot(gs[3, 0])
plot_time_domain(filtered_signal_hp, t, ax4, "Filtered Signal")

# 频域：原始信号
ax5 = fig.add_subplot(gs[0, 1])
plot_frequency_domain(stream, fs, ax5, "Original Signal (Freq)")

# 频域：调制信号
ax6 = fig.add_subplot(gs[1, 1])
plot_frequency_domain(modulated_signal, fs, ax6, "FM Modulated Signal (Freq)")

# 频域：添加噪声后的信号
ax7 = fig.add_subplot(gs[2, 1])
plot_frequency_domain(noisy_signal, fs, ax7, "FM Signal with Complex Noise (Freq)")

# 频域：滤波后的解调信号
ax8 = fig.add_subplot(gs[3, 1])
plot_frequency_domain(filtered_signal_hp, fs, ax8, "Filtered Signal After Downconversion (Freq)")


# 生成原信号频域和解调信号频域的图像
fig2 = plt.figure(figsize=(10, 6))  
ax9 = fig2.add_subplot(211)  
plot_frequency_domain(stream, fs, ax9, "Original Signal (Freq)")
ax9.set_xlim([-50, 50])


ax10 = fig2.add_subplot(212)
plot_frequency_domain(filtered_signal_hp, fs, ax10, "FM Demodulated Signal (Freq)")
ax10.set_xlim([-50, 50])

plt.tight_layout()  # 自动调整布局，防止子图重叠
plt.show()
