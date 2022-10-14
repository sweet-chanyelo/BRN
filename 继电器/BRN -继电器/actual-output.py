import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 导入标签数据
df = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
height, width = df.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df.iat[i, 0]
x = np.linspace(0, 183, 183)
plt.figure(1, figsize=(10, 4.5))
plt.plot(x, output, 'b--', lw=0.6, label='Actual state')
plt.scatter(x, output, color='', marker='v', edgecolors='b', s=12)
plt.ylim(0, 1.2)
font = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'normal', 'size': 12}  # 设置字体
font1 = {'family': 'Times New Roman',  'weight': 'normal', 'size': 12}  # 设置字体
plt.xlabel("Time of relay actions", font)       # x坐标轴
plt.ylabel("Actual state", font)  # y坐标轴
# plt.title('health state')
plt.xlim(0, 185)
plt.legend(prop=font1)
plt.grid(axis='y', linestyle='-.')
plt.show()
# size = np.random.rand(k) * 50
