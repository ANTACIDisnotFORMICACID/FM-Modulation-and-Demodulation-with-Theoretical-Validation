import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from matplotlib import gridspec

# FM调制函数
def fm_modulate(baseband, fs, f0, freq_shift):
    """
    改进的FM调制函数
    参数：
        baseband : 基带信号
        fs     : 采样率 (Hz)
        f0     : 载波频率 (Hz)
        freq_shift: 最大频率偏移 (Hz)
    返回：
        已调制的FM信号（实数）
    """
    # 计算调制信号的积分（归一化到[-1,1]）
    integrated = np.cumsum(baseband) / fs
    integrated /= np.max(np.abs(integrated))  # 归一化积分
    
    # 相位
    phase = 2 * np.pi * f0 * np.arange(len(baseband))/fs + 2 * np.pi * freq_shift * integrated
    return np.cos(phase)

# FM解调函数（希尔伯特变换）
def fm_demodulate(fm_signal, fs, f0, cutoff):
    """
    改进的FM解调函数
    参数：
        fm_signal : 接收到的FM信号（实数）
        fs     : 采样率 (Hz)
        f0     : 载波频率 (Hz)
        cutoff : 低通截止频率 (Hz)
    返回：
        解调后的基带信号
    """
    # 生成解析信号
    analytic_signal = hilbert(fm_signal)
    
    # 计算瞬时相位（解卷绕）
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
    # 计算瞬时频率（差分相位）
    instantaneous_freq = (np.diff(instantaneous_phase) * fs) / (2 * np.pi)
    
    # 去直流分量
    demodulated = instantaneous_freq - f0
    
    # 低通滤波
    return low_pass_filter(demodulated, fs, cutoff)

# 低通滤波器
def low_pass_filter(signal_data, fs, cutoff_freq, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal_data)

# 高通滤波器
def high_pass_filter(signal_data, fs, cutoff_freq, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal_data)

# 添加实部噪声
def add_noise(signal, snr_db):
    """
    添加高斯白噪声（实数）
    参数：
        signal : 输入信号
        snr_db : 信噪比 (dB)
    返回：
        加噪信号
    """
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    return signal + noise

# 绘图函数
def plot_time_domain(signal_data, t, ax, title):
    ax.plot(t, signal_data)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

def plot_frequency_domain(signal_data, fs, ax, title):
    N = len(signal_data)
    freq = np.fft.fftfreq(N, 1/fs)
    signal_freq = np.fft.fft(signal_data)
    signal_freq = np.fft.fftshift(signal_freq)
    freq = np.fft.fftshift(freq)
    magnitude = 20 * np.log10(np.abs(signal_freq)/np.max(np.abs(signal_freq)))  #dB
    ax.plot(freq, magnitude)
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True)
    return freq, magnitude

# 参数设置
fs = 4000 #采样频率
T = 2 #采样时间
t = np.arange(0, T, 1/fs)
f0 = 500 #载波频率
freq_shift = 50 #频率偏移量

# 生成基带信号
f1 = 5 #频率为5的正弦波
f2 = 10 #频率为10的正弦波
baseband = 0.5 * (np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)) #实例：两个正弦波叠加

# 信号处理流程

modulated = fm_modulate(baseband, fs, f0, freq_shift)

snr_db = 30 #添加一个信噪比为30的噪声
noisy_signal = add_noise(modulated, snr_db)

demodulated = fm_demodulate(noisy_signal, fs, f0, cutoff = f2) #PS:cutoff需>=baseband中最高频率(此处f2)

# 数组长度减少之后对齐处理
valid_idx = int(0.1*len(demodulated))
t_valid = t[valid_idx:-1]  # 时间轴对齐
demod_valid = demodulated[valid_idx:]

# 下变频处理
carrier = np.cos(2 * np.pi * f0 * t)
downconverted = noisy_signal * carrier
filtered_signal = low_pass_filter(downconverted, fs, 30)
filtered_signal_hp = high_pass_filter(filtered_signal, fs, 2)

# 绘图设置
fig = plt.figure(figsize=(24, 18))
gs = gridspec.GridSpec(5, 2, figure=fig)
gs.update(wspace=0.3, hspace=0.8)

# 时域图
ax1 = fig.add_subplot(gs[0, 0])
plot_time_domain(baseband, t, ax1, "Original Baseband Signal")

ax2 = fig.add_subplot(gs[1, 0])
plot_time_domain(modulated, t, ax2, "FM Modulated Signal")

ax3 = fig.add_subplot(gs[2, 0])
plot_time_domain(noisy_signal, t, ax3, "Noisy FM Signal")

ax4 = fig.add_subplot(gs[3, 0])
plot_time_domain(filtered_signal_hp, t, ax4, "Downconverted & Filtered Signal")

ax5 = fig.add_subplot(gs[4, 0])
plot_time_domain(demod_valid, t_valid, ax5, "Demodulated Signal")

# 频域图
ax6 = fig.add_subplot(gs[0, 1])
plot_frequency_domain(baseband, fs, ax6, "Baseband Spectrum")
ax6.set_xlim(-50, 50)

ax7 = fig.add_subplot(gs[1, 1])
plot_frequency_domain(modulated, fs, ax7, "FM Modulated Spectrum")
ax7.set_xlim(f0-100, f0+100)

ax8 = fig.add_subplot(gs[2, 1])
plot_frequency_domain(noisy_signal, fs, ax8, "Noisy FM Spectrum")
ax8.set_xlim(f0-100, f0+100)

ax9 = fig.add_subplot(gs[3, 1])
plot_frequency_domain(filtered_signal_hp, fs, ax9, "Filtered Spectrum")
ax9.set_xlim(-50, 50)

ax10 = fig.add_subplot(gs[4, 1])
plot_frequency_domain(demod_valid, fs, ax10, "Demodulated Spectrum")
ax10.set_xlim(-50, 50)

plt.tight_layout()
plt.show()
