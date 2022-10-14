"""

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_excel('输出结果.xlsx', 'data', header=None)
    df2 = pd.read_excel('BRN输出.xlsx', 'data', header=None)
    df3 = pd.read_excel('RANDOM输出.xlsx', 'data', header=None)
    raw, col = df1.shape
    BRN_PSO = np.zeros(raw - 1)
    BRN = np.zeros(raw - 1)
    RANDOM = np.zeros(raw - 1)
    for i in range(1, raw):
        BRN_PSO[i - 1] = df1.iloc[i]
        BRN[i - 1] = df2.iloc[i]
        RANDOM[i - 1] = df3.iloc[i]
        # 输出
    df4 = pd.read_excel('input2.xlsx', "output", header=None)
    row, col = df4.shape
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df4.iloc[i]
    # print(output)

    index = range(0, len(output), 5)
    train_y = np.delete(output, index, 0)
    test_y = output[index]

    fig1 = plt.figure(1)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10.5}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 12}  # 设置坐标轴字体
    plt.subplot(311)
    plt.plot(RANDOM, '--s', color='navy', lw=1, markersize='2.8', label='Random')
    plt.plot(test_y, '--v', color='r', lw=1, markersize='2.8', label='The actual status')
    plt.xlabel("Number of test (a)", font1)
    plt.ylabel("Result", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.legend(prop=font)

    #
    plt.subplot(312)
    plt.plot(BRN, '--s', color='g', lw=1, markersize='2.8', label='Expert')
    plt.plot(test_y, '--v', color='r', lw=1, markersize='2.8', label='The actual status')
    plt.xlabel("Number of test (b)", font1)
    plt.ylabel("Result", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.legend(prop=font)

    plt.subplot(313)
    plt.plot(BRN_PSO, '--s', color='blue', lw=1, markersize='2.8', label='BRN-PSO')
    plt.plot(test_y, '--v', color='r', lw=1, markersize='2.8', label='The actual status')
    plt.xlabel("Number of test (c)", font1)
    plt.ylabel("Result", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.legend(prop=font)

    fig1.tight_layout()
    plt.show()

    fig1.savefig('优化比较.svg', format='svg', dpi=250)