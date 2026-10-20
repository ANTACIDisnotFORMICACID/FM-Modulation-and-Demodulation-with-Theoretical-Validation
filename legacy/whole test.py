import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 采样率和信号参数
fs = 100e3  # 100 kHz 采样率
fm = 1e3  # 1 kHz 音频信号频率
fc = 10e3  # 10 kHz 载波频率
duration = 0.01  # 10ms 时长
kf = 50  # 调频指数

# 生成时间轴
t = np.arange(0, duration, 1/fs)

# 生成基带信号（音频）
audio_signal = np.sin(2 * np.pi * fm * t)

# FM 调制
integral_of_message = np.cumsum(audio_signal) / fs
fm_modulated = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral_of_message)

# 添加噪声
noise = 0.2 * np.random.randn(len(fm_modulated))
rx_signal = fm_modulated + noise

# FM 解调（使用差分法）
phase = np.unwrap(np.angle(rx_signal))
demodulated = np.diff(phase) * fs / (2 * np.pi * kf)

# 低通滤波，去除高频噪声
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

demodulated_filtered = butter_lowpass_filter(demodulated, fm * 2, fs)

# 画图
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, audio_signal, label="原始音频信号")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, fm_modulated, label="FM 调制信号")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t[:-1], demodulated_filtered, label="FM 解调信号")
plt.legend()

plt.tight_layout()
plt.show()
