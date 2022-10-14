import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
# ============================由excel导入样本数据=======================#
df = pd.read_excel('data3.xls', sheet_name='input-183', header=None)
# 样本量与数据维度
N, dim = df.shape
# 特征两两不重复组合
input = np.zeros((N - 1, dim))
for i in range(1, N):
    for j in range(0, dim):
        input[i - 1][j] = df.iat[i, j]
x = np.linspace(0, 183, 183)
X, y = make_classification(n_samples=183, n_features=10, n_classes=5, n_informative=4, random_state=0)
R0 = input[:, 0]
V = input[:, 1]
R1 = input[:, 2]
R2 = input[:, 3]
T1 = input[:, 4]
T2 = input[:, 5]

font = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 15}  # 设置坐标轴字体
font1 = {'family': 'Times New Roman', 'size': 12}  # 设置字体

plt.figure(figsize=(4.8, 4))
plt.hist(x, 183, normed='R0', edgecolor='k', facecolor='g', alpha=0.5, label='Resistance of winding(R0)')  # 直方图
# plt.scatter(x, R0, marker='o', c=y, cmap='summer', s=24, label='Resistance of winding(R0)')  # 散点图
plt.grid(axis='y', linestyle='-.')
m = range(0, 183, 30)
plt.xticks(m, family='Times New Roman', fontsize=7)
plt.yticks(family='Times New Roman', fontsize=7)
plt.xlabel("Times of relay actions", font)       # x坐标轴
plt.ylabel("Resistance of winding/$Ω$", font)    # y坐标轴
plt.legend(prop=font1)  # 显示图例


plt.figure(figsize=(4.7, 4))
plt.scatter(x, V, marker='o', c=y, cmap='Oranges', s=24, label='Voltage of pull-in(V)')
plt.grid(axis='y', linestyle='-.')
plt.xticks(m, family='Times New Roman', fontsize=7)
plt.yticks(family='Times New Roman', fontsize=7)
plt.ylim(13.9, 14.7)   # y幅度
plt.xlabel("Times of relay actions", font)       # x坐标轴
plt.ylabel("Voltage of pull-in/V", font)    # y坐标轴
plt.legend(prop=font1)  # 显示图例

plt.figure(figsize=(4.7, 4))
plt.scatter(x, R1, marker='o', c=y, cmap='plasma', s=24, label='Resistance of open point(R1)')
plt.grid(axis='y', linestyle='-.')
plt.xticks(m, family='Times New Roman', fontsize=7)
plt.yticks(family='Times New Roman', fontsize=7)
plt.xlabel("Times of relay actions", font)       # x坐标轴
plt.ylabel("Resistance of open point/m$Ω$", font)    # y坐标轴
plt.legend(prop=font1)  # 显示图例

plt.figure(figsize=(4.7, 4))
plt.scatter(x, R2, marker='o', c=y, cmap='Set1', s=24, label='Resistance of closed point(R2)')
plt.grid(axis='y', linestyle='-.')
plt.xticks(m, family='Times New Roman', fontsize=7)
plt.yticks(family='Times New Roman', fontsize=7)
plt.ylim(42.8, 44.2)   # y幅度
plt.xlabel("Times of relay actions", font)       # x坐标轴
plt.ylabel("Resistance of closed point/m$Ω$", font)    # y坐标轴
plt.legend(prop=font1)  # 显示图例

plt.figure(figsize=(4.7, 4))
plt.scatter(x, T1, marker='o', c=y, cmap='rainbow', s=24, label='Time of release disconnect(T0)')
plt.grid(axis='y', linestyle='-.')
plt.xticks(m, family='Times New Roman', fontsize=7)
plt.yticks(family='Times New Roman', fontsize=7)
plt.xlabel("Times of relay actions", font)       # x坐标轴
plt.ylabel("Time of release disconnect/ms", font)    # y坐标轴
plt.ylim(1.405, 1.425)   # y幅度
plt.legend(prop=font1)  # 显示图例

plt.figure(figsize=(4.7, 4))
plt.scatter(x, T2, marker='o', c=y, cmap='Spectral', s=24, label='Time of release(T1)')
plt.grid(axis='y', linestyle='-.')
plt.xticks(m, family='Times New Roman', fontsize=7)
plt.yticks(family='Times New Roman', fontsize=7)
plt.ylim(1.615, 1.638)   # y幅度
plt.xlabel("Times of relay actions", font)       # x坐标轴
plt.ylabel("Time of release bounce back/ms", font)    # y坐标轴
plt.legend(prop=font1)  # 显示图例
plt.show()

