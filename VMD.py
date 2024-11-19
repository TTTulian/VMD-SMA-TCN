import pandas as pd
from vmdpy import VMD
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

data = pd.read_excel("四种负荷潜力.xlsx")
#GY3、CZ1、XC2、NY4
f = np.array(data['1u'])
# f = f[1920:3920]

alpha = 2000       # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-6
K=7

u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
# VMD分解
# for K in range(3, 10):
#     u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

modes = []

for i in range(K):
    mode_i = u[i, :] * np.resize(omega[i], f.shape)
    modes.append(mode_i)

reconstructed_signal = np.sum(modes, axis=0)

# 计算重构信号与原始信号之间的RMSE值
# def calculate_rmse(original_signal, reconstructed_signal):
#     return np.sqrt(np.mean((original_signal - reconstructed_signal)**2))
#
# rmse = calculate_rmse(f, reconstructed_signal)
# print('K=', K, ', RMSE:', rmse)

# 计算中心频率
delta_t = 1  # 时间间隔
Fs = 1/delta_t  # 采样频率
N = len(f)  # 信号长度
t = np.arange(N) * delta_t  # 时间序列
freqs = np.fft.fftfreq(N, delta_t)  # 频率序列
spectrum = np.fft.fft(reconstructed_signal)  # 频谱

positive_freqs = freqs[:N // 2]  # 取正频率部分
positive_spectrum = np.abs(spectrum[:N // 2])  # 取正频率部分的幅度

center_frequency = np.sum(positive_spectrum * positive_freqs) / np.sum(positive_spectrum)
print('K=', K, ', Center frequency:', center_frequency)

# 输出每个模态的中心频率
for i in range(K):
    mode_i_spectrum = np.fft.fft(modes[i])
    mode_i_positive_spectrum = np.abs(mode_i_spectrum[:N // 2])
    mode_i_center_frequency = np.sum(mode_i_positive_spectrum * positive_freqs) / np.sum(mode_i_positive_spectrum)
    print('Mode', i + 1, 'center frequency:', mode_i_center_frequency)

# # 绘图
# fig, axs = plt.subplots(K+1, figsize=(8, 12))
#
# axs[0].plot(f, color='blue')
# axs[0].set_ylabel('原始序列（MW）')
# axs[0].set_title('上调潜力分解')
#
# # colors = ['red', 'green', 'orange', 'yellow','purple'], color=colors[i]
# for i in range(K):
#     axs[i+1].plot(u[i, :])
#     axs[i+1].set_ylabel('IMF-{}'.format(i+1))
#
# axs[K].set_xlabel('采样时间')
# plt.tight_layout()
# import numpy as np
# np.set_printoptions(threshold=np.inf)
# print(omega)


fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10,4.8))

d=u[0, :]+u[1, :]
z=u[2, :]+u[3, :]
g=u[4, :]+u[5, :]+u[6, :]

ax1.plot(d)
ax1.set_ylabel('低频分量')
ax1.set_xlabel('采样时间')
ax2.plot(z)
ax2.set_ylabel('中频分量')
ax2.set_xlabel('采样时间')
ax3.plot(g)
ax3.set_ylabel('高频分量')
ax3.set_xlabel('采样时间')

VMD1 = pd.DataFrame(d)
VMD1.to_excel(excel_writer="1u-低频.xlsx")

VMD2 = pd.DataFrame(z)
VMD2.to_excel(excel_writer="1u-中频.xlsx")

VMD3 = pd.DataFrame(g)
VMD3.to_excel(excel_writer="1u-高频.xlsx")













plt.show()
