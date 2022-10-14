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

font = {'family': 'Times New Roman', 'style': 'oblique',  'size': 10}  # 设置坐标轴字体
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}  # 设置字体

plt.subplots(15, 10, sharex='all')  # 共享x轴
ax1 = plt.subplot(6, 1, 1)
plt.subplots_adjust(hspace=0.22)   # 取消空隙

# 第一条线
plt.subplot(611)
plt.plot(x, R0,  'b.', linewidth=0.4, label='Resistance of winding(RW)')  # 散点图
# plt.setp(a1.get_xticklabels(), fontsize=6)  # 隐藏坐标轴
plt.ylabel("RW($Ω$)", font)    # y坐标轴
plt.grid(axis='y', linestyle='-.')  # 显示背景y轴
plt.legend(prop=font1)  # 显示图例

# plt.ylim()
# 第二条线
plt.subplot(612)
plt.plot(x, V,  'b.', linewidth=0.4, label='Voltage of pull-in(VPI)')  # 散点图
plt.ylabel("VPI(V)", font)    # y坐标轴
plt.ylim(13.9, 14.6)   # y幅度
plt.yticks([14.00, 14.50])  # 设置y坐标轴刻度
plt.grid(axis='y', linestyle='-.')
plt.legend(prop=font1)  # 显示图例

# 第三条线
plt.subplot(613)
plt.plot(x, R1,  'b.', linewidth=0.4, label='Resistance of open point(ROC)')  # 散点图
plt.ylabel("ROC(m$Ω$)", font)    # y坐标轴
plt.grid(axis='y', linestyle='-.')
plt.legend(prop=font1)  # 显示图例

# 第四条线
plt.subplot(614)
plt.plot(x, R2,  'b.', linewidth=0.4, label='Resistance of close point(RCC)')  # 散点图
plt.ylabel("RCC(m$Ω$)", font)    # y坐标轴
plt.ylim(42.5, 44.5)   # y幅度
plt.yticks([43.0, 44.0], [43.0, 44.0])  # 设置y坐标轴刻度
plt.grid(axis='y', linestyle='-.')
plt.legend(prop=font1)  # 显示图例

# 第五条线
plt.subplot(615)
plt.plot(x, T1,  'b.', linewidth=0.4, label='Time of release disconnect(TRD)')  # 散点图
plt.ylabel("TRD(ms)", font)    # y坐标轴
plt.ylim(1.405, 1.425)   # y幅度
plt.grid(axis='y', linestyle='-.')
plt.legend(prop=font1)  # 显示图例

# 第六条线
plt.subplot(616)
plt.plot(x, T2,  'b.', linewidth=0.4, label='Time of release bounce back(TRBB)')  # 散点图
plt.ylabel("TRBB(ms)", font)    # y坐标轴
plt.ylim(1.615, 1.638)   # y幅度
plt.grid(axis='y', linestyle='-.')
plt.legend(prop=font1)  # 显示图例

plt.xticks(family='Times New Roman', fontsize=12)
plt.xlabel("Times of relay actions", font)       # x坐标轴

plt.show()