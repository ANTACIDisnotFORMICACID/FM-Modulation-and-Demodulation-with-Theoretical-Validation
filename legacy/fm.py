#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:36:56 2025

@author: huangminyu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# 基础参数
fs=1000
t=np.arange(0,1,1/fs)
f_m=10
f_c=100
Kf=500
A=1

# 基带信号
baseband_signal=np.cos(2*np.pi*f_m*t) + np.cos(2*np.pi*15*t)

# 积分器
integral_signal=baseband_signal.copy()/fs
for i in range(1,fs):
    integral_signal[i]=integral_signal[i]+integral_signal[i-1]
    
# 载波信号
carrier_signal=A*np.cos(2*np.pi*f_c*t)


# 正交调制
I=np.cos(Kf*integral_signal)
Q=np.sin(Kf*integral_signal)
fm_signal=I*carrier_signal-Q*A*np.sin(2*np.pi*f_c*t)





# 微分器
differ_signal=fm_signal.copy()
for i in range(fs-1):
    differ_signal[i]=(differ_signal[i+1]-differ_signal[i])*fs
    
# 全波整流
rectified = np.abs(differ_signal)

# 设计低通滤波器参数
cutoff = 2 * f_m  # 截止频率设为调制频率的两倍
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(4, normal_cutoff, btype='low')  # 4阶滤波器

# 滤波
filter_signal = filtfilt(b, a, rectified)

# 去除直流
sum1=0
for i in range(fs):
    sum1=sum1+filter_signal[i]
    
sum1=sum1/fs

filter_signal=filter_signal-sum1
    
# 放大缩小
sum2=0
sum3=0
rec_baseband=np.abs(baseband_signal)
rec_filter=np.abs(filter_signal)
for i in range(fs):
    sum2=sum2+rec_baseband[i]
    sum3=sum3+rec_filter[i]

filter_signal=filter_signal*(sum2/sum3)


fig, axs = plt.subplots(6, 1, figsize=(12, 8))

# 基带信号图例
axs[0].plot(t, baseband_signal)
axs[0].set_title('Baseband Signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].grid()

# 积分信号图例
axs[1].plot(t, integral_signal)
axs[1].set_title('Integral Signal')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')
axs[1].grid()

# 载波信号图例
axs[2].plot(t, carrier_signal)
axs[2].set_title('Carrier Signal')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Amplitude')
axs[2].grid()

# FM 信号图例
axs[3].plot(t, fm_signal)
axs[3].set_title('FM Signal')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Amplitude')
axs[3].grid()

# 微分信号图例
axs[4].plot(t, differ_signal)
axs[4].set_title('Differ Signal')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('Amplitude')
axs[4].grid()

# 滤波信号图例
axs[5].plot(t, filter_signal)
axs[5].set_title('Filter Signal')
axs[5].set_xlabel('Time (s)')
axs[5].set_ylabel('Amplitude')
axs[5].grid()


plt.tight_layout()
plt.savefig('fm.png')
plt.show()
