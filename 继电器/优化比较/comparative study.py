import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 导入标签数据
df1 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
height, width = df1.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df1.iat[i, 0]
label_data = []
for i in range(0, int(height / 5)):
    label_data.append(output[5 * i])
# 画图
x = np.linspace(0, 36, 36)
fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
ax.plot(x, label_data, 'k--', lw=1.2, label=' The actual state')
ax.scatter(x, label_data, color='', marker='v', edgecolors='k', s=24)
# --------------------------------------BRN0------------------------------------------#
df2 = pd.read_excel('brn0.xls', sheet_name='brn0', header=None)
height, width = df2.shape
output0 = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output0[i - 1] = df2.iat[i, 0]
brn0_data = []
for i in range(0, int(height / 5)):
    brn0_data.append(output0[5 * i])
# 画图
ax.plot(x, brn0_data, 'b--', lw=1.2, label='The state assessment by BRN0')
ax.scatter(x, brn0_data, color='', marker='p', edgecolors='b', s=24)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%BRN1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df3 = pd.read_excel('brn1.xls', sheet_name='brn1', header=None)
height, width = df3.shape
output1 = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output1[i - 1] = df3.iat[i, 0]
brn1_data = []
for i in range(0, int(height / 5)):
    brn1_data.append(output1[5 * i])
# 画图
ax.plot(x, brn1_data, 'm--', lw=1.2, label='The state assessment by BRN1')
ax.scatter(x, brn1_data, color='', marker='p', edgecolors='m', s=24)
# *****************************************************************************
df4 = pd.read_excel('brn2.xls', sheet_name='brn2', header=None)
height, width = df4.shape
output2 = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output2[i - 1] = df4.iat[i, 0]
brn2_data = []
for i in range(0, int(height / 5)):
    brn2_data.append(output2[5 * i])
# 画图
ax.plot(x, brn2_data, 'g--', lw=1.2, label='The state assessment by BRN2')
ax.scatter(x, brn2_data, color='', marker='p', edgecolors='g', s=24)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
df5 = pd.read_excel('brn3.xls', sheet_name='brn3', header=None)
height, width = df5.shape
output3 = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output3[i - 1] = df5.iat[i, 0]
brn3_data = []
for i in range(0, int(height / 5)):
    brn3_data.append(output3[5 * i])
# 画图
ax.plot(x, brn3_data, 'r--', lw=1.2, label='The state assessment by BRN3')
ax.scatter(x, brn3_data, color='', marker='p', edgecolors='r', s=24)


plt.xlim(-0.5, 38)            # x幅度
plt.ylim(0, 1.2)           # y幅度
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置字体
font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 15}  # 设置坐标轴字体
m = range(0, 38, 2)
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.xlabel("Time of relay actions", font1)       # x坐标轴
plt.ylabel("State assessment result", font1)  # y坐标轴
# plt.title('health state')
plt.legend(prop=font)
plt.grid(axis='y', linestyle='-.')
plt.show()