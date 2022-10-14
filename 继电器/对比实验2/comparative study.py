import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 导入标签数据
df0 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
height, width = df0.shape
output = np.zeros(height - 1)  # 默认为只有1列（单输出）
for i in range(1, height):
    output[i - 1] = df0.iat[i, 0]
label_data = []
for i in range(0, 25):
    label_data.append(output[5 * i])
for i in range(0, 11):
    # 测试数据
    label_data.append(output[125 + 5 * i + 1])
    label_data.append(output[125 + 5 * i + 2])
    label_data.append(output[125 + 5 * i + 3])
    label_data.append(output[125 + 5 * i + 4])

# 画图
x = np.linspace(0, 69, 69)
fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
ax.plot(x, label_data, 'r--', lw=1.2, label=' The actual state')
ax.scatter(x, label_data, color='', marker='v', edgecolors='r', s=24)
# --------------------------------------BRN------------------------------------------#
df1 = pd.read_excel('brn.xls', sheet_name='brn', header=None)
height, width = df1.shape
output = np.zeros(69)  # 默认为只有1列（单输出）
for i in range(1, 70):
    output[i - 1] = df1.iat[i, 0]
brn_data = output
# 画图
ax.plot(x, brn_data, 'b--', lw=1.2, label='The state assessment by BRN')
ax.scatter(x, brn_data, color='', marker='p', edgecolors='b', s=24)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%BP神经网络%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df2 = pd.read_excel('brnn2.xls', sheet_name='BPNN2', header=None)
height, width = df2.shape
output = np.zeros(69)  # 默认为只有1列（单输出）
for i in range(1, 70):
    output[i - 1] = df2.iat[i, 0]
bpnn_data = output
# 画图
ax.plot(x, bpnn_data, 'g--', lw=1, label='The state assessment by BPNN')
ax.scatter(x, bpnn_data, color='', marker='o', edgecolors='g', s=24)

# ************************************ SVM *****************************************
df3 = pd.read_excel('svm.xls', sheet_name='svm', header=None)
height, width = df3.shape
output = np.zeros(69)  # 默认为只有1列（单输出）
for i in range(1, 70):
    output[i - 1] = df3.iat[i, 0]
svm_data = output
# 画图
ax.plot(x, svm_data, 'k--', lw=1, label='The state assessment by SVM')
ax.scatter(x, svm_data, color='', marker='s', edgecolors='k', s=24)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 画图 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
plt.xlim(-0.5, 70)            # x幅度
plt.ylim(0, 1.2)           # y幅度
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置字体
font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 15}  # 设置坐标轴字体
m = range(0, 70, 5)
plt.xticks(m, family='Times New Roman', fontsize=12)
plt.yticks(family='Times New Roman')
plt.xlabel("Times of relay actions", font1)       # x坐标轴
plt.ylabel("State assessment result", font1)  # y坐标轴
# plt.title('health state')
plt.legend(prop=font)
plt.grid(axis='y', linestyle='-.')

# 画图
plt.figure(2)
err = 0
error = np.zeros(len(label_data))
for i in range(0, len(label_data)):
    err = err + (brn_data[i] - label_data[i]) ** 2
    error[i] = err / (i + 1)
plt.plot(x, error, 'bp--', label='The error change process of BRN')

err = 0
for i in range(0, len(label_data)):
    err = err + (bpnn_data[i] - label_data[i]) ** 2
    error[i] = err / (i + 1)
plt.plot(x, error, 'go--', label='The error change process of BPNN')

err = 0
for i in range(0, len(label_data)):
    err = err + (svm_data[i] - label_data[i]) ** 2
    error[i] = err / (i + 1)
plt.plot(x, error, 'ks--', label='The error change process of SVM')

plt.xticks(m, family='Times New Roman', fontsize=12)
plt.yticks(family='Times New Roman')
plt.xlabel("Times of relay actions", font1)       # x坐标轴
plt.ylabel("Error change process", font1)  # y坐标轴
# plt.title('health state')
plt.legend(prop=font)
plt.grid(axis='y', linestyle='-.')

plt.show()