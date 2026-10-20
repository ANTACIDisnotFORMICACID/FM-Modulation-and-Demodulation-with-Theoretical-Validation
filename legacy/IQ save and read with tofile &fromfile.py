import numpy as np
import struct

# 保存复数信号到IQ文件的函数
"""def save_iq_file(file_path, iq_samples):
    
    将复数信号保存到 IQ 文件（每个复数由实部和虚部组成）
    file_path: 输出文件路径
    iq_samples: 复数信号数组（例如 numpy array）
    
    with open(file_path, 'wb') as f:
        for sample in iq_samples:
            # 提取复数的实部和虚部
            i_sample = np.int16(np.real(sample))  # 实部
            q_sample = np.int16(np.imag(sample))  # 虚部
            
            # 将实部和虚部按小端方式写入文件
            f.write(struct.pack('<hh', i_sample, q_sample))  # 写入实部和虚部，格式为小端16位整数

"""
# 创建复数信号数组
iq_samples = np.array([0+0j, 0+1j, 1+0j, 1+1j])

# 保存到 IQ 文件
output_file = 'output.iq'
"""
save_iq_file(output_file, iq_samples)
"""
print(type(iq_samples[0]))
iq_samples=iq_samples.astype(np.complex64) #切换格式使之与.iq兼容
print(type(iq_samples[0]))
iq_samples.tofile(output_file)

print(f"IQ信号已保存到 {output_file}")

# 读取IQ文件的函数
"""
def read_iq_file(file_path):
    
    读取IQ文件并返回复数信号数组
    :param file_path: 输入文件路径
    :return: 复数信号（实部+虚部）
    
    iq_samples = []
    with open(file_path, 'rb') as f:
        while True:
            # 每次读取4个字节（两个16位整数）
            data = f.read(4)
            if not data:
                break
            # 解析实部和虚部
            i_sample, q_sample = struct.unpack('<hh', data)
            iq_samples.append(i_sample + 1j * q_sample)
    
    return np.array(iq_samples)
"""

# 读取IQ文件
"""
iq_data = read_iq_file('output.iq')
"""

iq_data = np.fromfile(output_file, np.complex64)#注意p ython默认复数格式为complex128以合适格式读取

# 打印读取的数据
print(iq_data) #此处输出浮点数会有0.的格式，可为美观格式化为整数

# 转换为整数
# 示例复数数组
iq_data = np.array([0.+0.j, 0.+1.j, 1.+0.j, 1.+1.j])

# 格式化输出
formatted_iq_data = [f"{x.real:.0f}+{x.imag:.0f}j" for x in iq_data]
print("格式化后的IQ信号:", formatted_iq_data)
