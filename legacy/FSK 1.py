import numpy as np
import matplotlib.pyplot as plt

# BFSK调制函数
def fsk_modulate(bitstream, f0, f1, fs, T_bit):
    """
    用BFSK进行调制，并确保相位连续
    :param bitstream: 输入的二进制比特流（如[1, 0, 1, 1, 0, 1]）
    :param f0: 比特0对应的频率
    :param f1: 比特1对应的频率
    :param fs: 采样频率
    :param T_bit: 每个比特的持续时间
    :return: 调制后的信号
    """
    t = np.arange(0, T_bit * len(bitstream), 1/fs)  # 生成时间向量
    signal = np.array([])  # 存储调制后的信号
    phase = 0  # 初始相位

    # 对每一个比特进行调制
    for bit in bitstream:
        # 根据比特选择频率
        if bit == 0:
            freq = f0
        else:
            freq = f1
        
        # 生成每个比特对应的时间向量
        t_bit = np.arange(0, T_bit, 1/fs)  # 每个比特的时间向量
        phase_increment = 2 * np.pi * freq * t_bit  # 单位时间内的相位变化量

        # 生成载波信号
        carrier = np.cos(phase + phase_increment)  # 使用累加的相位生成正弦波

        # 累加相位，保持连续性，并避免反相
        phase += phase_increment[-1]  # 只取最后一个相位增量

        # 将载波信号添加到总信号中
        signal = np.concatenate((signal, carrier))
    
    return signal, t[:len(signal)]  # 返回调制后的信号和时间向量

# 参数设置
bitstream = [1, 0, 1, 1, 0, 1]  # 输入比特流
f0 = 15  # 比特0对应的频率（Hz）
f1 = 25  # 比特1对应的频率（Hz）
fs = 1000  # 采样频率（Hz）
T_bit = 0.1  # 每个比特持续时间（秒）

# 调制
modulated_signal, t_signal = fsk_modulate(bitstream, f0, f1, fs, T_bit)

# 绘制调制后的信号
plt.figure(figsize=(10, 4))
plt.plot(t_signal, modulated_signal)
plt.title("BFSK Modulated Signal with Continuous Phase")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
