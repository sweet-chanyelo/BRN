from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 导入标签数据
df1 = pd.read_excel('data4.xls', sheet_name='output-183', header=None)
height, width = df1.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df1.iat[i, 0]
label_data = []
for i in range(0, int(height / 5)):
    label_data.append(output[5 * i])
# 画图
x = np.linspace(0, int(height / 5), int(height / 5))
fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
ax.plot(x, label_data, 'r--', lw=1.2, label='The actual state')
ax.scatter(x, label_data, color='', marker='v', edgecolors='r', s=24)

# --------------------------------------BRN------------------------------------------#
df3 = pd.read_excel('brn.xls', sheet_name='brn', header=None)
height, width = df3.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df3.iat[i, 0]
brn_data = []
for i in range(0, int(height / 5)):
    brn_data.append(output[5 * i])
# 画图
ax.plot(x, brn_data, 'b--', lw=1.2, label='The state assessment by BRN1')
ax.scatter(x, brn_data, color='', marker='p', edgecolors='b', s=24)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%BP神经网络%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df2 = pd.read_excel('bpnn.xls', sheet_name='BPNN', header=None)
height, width = df2.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df2.iat[i, 0]
bpnn_data = output
# 画图
ax.plot(x, bpnn_data, 'g--', lw=1, label='The state assessment by BPNN')
ax.scatter(x, bpnn_data, color='', marker='o', edgecolors='g', s=24)
# *****************************************************************************
df4 = pd.read_excel('brn2.xls', sheet_name='brn2', header=None)
height, width = df4.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df4.iat[i, 0]
brn2_data = []
for i in range(0, int(height / 5)):
    brn2_data.append(output[5 * i])
# 画图
ax.plot(x, brn2_data, 'k--', lw=1, label='The state assessment by BRN0')
ax.scatter(x, brn2_data, color='', marker='s', edgecolors='k', s=24)
# ===============================BRB====================================================
df4 = pd.read_excel('trbrb.xls', sheet_name='trbrb', header=None)
height, width = df4.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df4.iat[i, 0]
brb_data = []
for i in range(0, int(height / 5)):
    brb_data.append(output[5 * i])
# 画图
y = 100
ax.plot(x, brb_data, 'y--', lw=1, label='The state assessment by BRB')
ax.scatter(x, brb_data, color='', marker='d', edgecolors='y', s=24)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
plt.ylim(0, 1.2)           # y幅度
font = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'normal', 'size': 15}  # 设置字体
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置字体
plt.xlabel("Time of relay actions", font)       # x坐标轴
plt.ylabel("State assessment result", font)  # y坐标轴
# plt.title('health state')
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.legend(prop=font1)
plt.grid(axis='y', linestyle='-.')
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&局部放大&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7
axins = inset_axes(ax, width="5%", height='20%', loc='lower left', bbox_to_anchor=(0.46, 0.55, 1.5, 1.5), bbox_transform=ax.transAxes)
axins.plot(x, label_data, 'r--', lw=1.2, label=' The actual fault state')
axins.scatter(x, label_data, color='', marker='v', edgecolors='r', s=25)
axins.plot(x, brn_data, 'b--', lw=1.2, label=' The BRN fault state')
axins.scatter(x, brn_data, color='', marker='p', edgecolors='b', s=25)
axins.plot(x, bpnn_data, 'g--', lw=1, label=' The BP fault state')
axins.scatter(x, bpnn_data, color='', marker='o', edgecolors='g', s=24)
axins.plot(x, brb_data, 'y--', lw=1, label=' The BRB fault state')
axins.scatter(x, brb_data, color='', marker='d', edgecolors='y', s=24)
axins.plot(x, brn2_data, 'k--', lw=1, label=' The BRN without optimize fault state')
axins.scatter(x, brn2_data, color='', marker='s', edgecolors='k', s=24)
# 设置放大区间
zone_left = 14
zone_right = 15
# 坐标轴的扩展比例
x_ratio = 0.8
y_ratio = 0.5
# X轴的显示范围
xlim0 = 13.7
xlim1 = 15.4
# Y轴的显示范围
y = np.hstack((label_data[zone_left:zone_right], brn_data[zone_left:zone_right], bpnn_data[zone_left:zone_right], brb_data[zone_left:zone_right], brn2_data[zone_left:zone_right]))
ylim0 = 0.3
ylim1 = 0.5
# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='k', lw=0.6)

plt.show()