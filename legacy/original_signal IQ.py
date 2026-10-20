import numpy as np

fs = 400
T = 2
A = 1
t = np.arange(0, T, 1/fs)
original = np.sin(2 * np.pi * 5 * t)

output_file = 'original.iq'
print(type(original[0]))
original=original.astype(np.complex64) #切换格式使之与.iq兼容
print(type(original[0]))
original.tofile(output_file)

print(f"IQ信号已保存到 {output_file}")
