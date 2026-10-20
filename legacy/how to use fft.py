import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

Fs=1 #Hz
N=100 #采样点数
f=0.15 #Hz原频率

t=np.arange(0,N/Fs,1/Fs)
s=np.sin(2*np.pi*t*f) #复数信号
s=s*np.hamming(100) #窗函数：保持信号片段的循环特性
S=np.fft.fftshift(np.fft.fft(s))

S_mag=np.abs(S)
S_phase=np.angle(S)
f=np.arange(Fs/-2,Fs/2,Fs/N)

plt.figure(0)
plt.plot(f, S_mag,'.-')
plt.figure(1)
plt.plot(f, S_phase,'.-')
plt.show()
