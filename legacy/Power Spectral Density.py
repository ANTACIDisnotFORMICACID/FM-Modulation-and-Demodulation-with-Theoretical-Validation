import numpy as np
import matplotlib.pyplot as plt

Fs = 300 # 采样率
Ts = 1/Fs # 采样间隔
N = 2048 # 用于模拟的样本数

t = Ts*np.arange(N)
x = np.exp(1j*2*np.pi*50*t) # 模拟一个 50 Hz 的正弦信号

n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # 单位功率的复信号噪声
noise_power = 2
r = x + n * np.sqrt(noise_power)

PSD = np.abs(np.fft.fft(r))**2 / (N*Fs)
PSD_log = 10.0*np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.arange(Fs/-2.0, Fs/2.0, Fs/N) # 起始值，结束值，步长

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()
