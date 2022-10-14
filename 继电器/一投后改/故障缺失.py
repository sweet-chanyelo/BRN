"""
CART>> RF>> BRN-PSO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_excel('CART故障缺失.xlsx', 'data', header=None)
    df2 = pd.read_excel('RF故障缺失.xlsx', 'data', header=None)
    df3 = pd.read_excel('BRN故障缺失.xlsx', 'data', header=None)
    df4 = pd.read_excel('RIMER-PCA故障缺失.xlsx', 'data', header=None)
    raw, col = df1.shape
    CART = np.zeros(raw - 1)
    RF = np.zeros(raw - 1)
    BRN = np.zeros(raw - 1)
    RIMER_PCA = np.zeros(raw - 1)
    for i in range(1, raw):
        CART[i - 1] = df1.iloc[i]
        RF[i - 1] = df2.iloc[i]
        BRN[i - 1] = df3.iloc[i]
        RIMER_PCA[i - 1] = df4.iloc[i]
    # df5 = pd.read_excel('测试输出.xlsx', 'data2', header=None)
    df6 = pd.read_excel('input2.xlsx', 'output', header=None)
    raw, col = df6.shape
    output = np.zeros(raw - 1)
    for i in range(1, raw):
        output[i - 1] = df6.iloc[i]
    # print(len(output))
    # 画图
    # y = np.linspace(1, 44, 44)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 11}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 12}  # 设置坐标轴字体
    fig1 = plt.figure(figsize=(8, 7))
    plt.rcParams['xtick.direction'] = 'in'  # X轴下标朝内
    plt.rcParams['ytick.direction'] = 'in'  # Y轴下标朝内

    ax1 = plt.subplot(411)
    ax1.plot(CART, '--s', color='b', lw=1, markersize='3', label='CART')
    ax1.plot(output, '--h', color='r', lw=1, markersize='2.5', label='The actual status')
    # ax1.plot([170, 170], [1, 0], 'k-.', lw=1.5)
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax1.legend(prop=font)

    ax2 = plt.subplot(412)
    ax2.plot(RF, '-->', lw=1, markersize='3', label='RF')
    ax2.plot(output, '--h', color='r', lw=1, markersize='2.5', label='The actual status')
    # ax2.plot([170, 170], [1, 0], 'k-.', lw=1.5)
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax2.legend(prop=font)

    ax3 = plt.subplot(413)
    ax3.plot(BRN, '--*', color='g', lw=1, markersize='3', label='BRN-PSO')
    ax3.plot(output, '--h', color='r', lw=1, markersize='2.5', label='The actual status')
    # ax3.plot([170, 170], [1, 0], 'k-.', lw=1.5)
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax3.legend(prop=font)

    ax4 = plt.subplot(414)
    ax4.plot(RIMER_PCA, '--p', color='navy', lw=1, markersize='3', label='RIMER_PCA')
    ax4.plot(output, '--h', color='r', lw=1, markersize='2.5', label='The actual status')
    # ax4.plot([170, 170], [1, 0], 'k-.', lw=1.5)
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax4.legend(prop=font)

    fig1.tight_layout()
    plt.show()
    fig1.savefig('故障缺失对比实验.svg', format='svg', dpi=250)