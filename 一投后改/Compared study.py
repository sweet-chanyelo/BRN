"""
RF >> SVM >> KNN >> BPNN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


if __name__ == '__main__':
    df1 = pd.read_excel('BPNN.xlsx', 'data', header=None)
    df2 = pd.read_excel('SVM.xlsx', 'data', header=None)
    df3 = pd.read_excel('KNN.xlsx', 'data', header=None)
    df4 = pd.read_excel('RF.xlsx', 'data', header=None)
    df5 = pd.read_excel('输出结果.xlsx', 'data', header=None)
    df7 = pd.read_excel('GMDH.xlsx', 'data', header=None)
    raw, col = df1.shape
    BPNN = np.zeros(raw - 1)
    SVM = np.zeros(raw - 1)
    KNN = np.zeros(raw - 1)
    RF = np.zeros(raw - 1)
    BRN = np.zeros(raw - 1)
    GMDH = np.zeros(raw - 1)
    for i in range(1, raw):
        BPNN[i - 1] = df1.iloc[i]
        SVM[i - 1] = df2.iloc[i]
        KNN[i - 1] = df3.iloc[i]
        RF[i - 1] = df4.iloc[i]
        BRN[i - 1] = df5.iloc[i]
        GMDH[i - 1] = df7.iloc[i]
    # df5 = pd.read_excel('测试输出.xlsx', 'data2', header=None)
    df6 = pd.read_excel('input2.xlsx', 'output', header=None)
    raw, col = df6.shape
    output = np.zeros(raw - 1)
    for i in range(1, raw):
        output[i - 1] = df6.iloc[i]
    # print(len(output))

    index = range(0, len(output), 5)
    train_y = np.delete(output, index, 0)
    test_y = output[index]

    # 画图
    y = np.linspace(1, 44, 44)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 11}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 12}  # 设置坐标轴字体
    fig1 = plt.figure(figsize=(10, 6))
    plt.rcParams['xtick.direction'] = 'in'  # X轴下标朝内
    plt.rcParams['ytick.direction'] = 'in'  # Y轴下标朝内

    ax1 = plt.subplot(231)
    ax1.plot(y, BRN, '--s', color='b', lw=1, markersize='4', label='BRN-PSO')
    ax1.plot(y, test_y, '--h', color='r', lw=1, markersize='4', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax1.legend(prop=font)

    ax2 = plt.subplot(232)
    ax2.plot(y, BPNN, '-->', lw=1, markersize='4', label='BPNN-15')
    ax2.plot(y, test_y, '--h', color='r', lw=1, markersize='4', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax2.legend(prop=font)

    ax3 = plt.subplot(233)
    ax3.plot(y, GMDH, '--*', color='g', lw=1, markersize='4', label='GMDH')
    ax3.plot(y, test_y, '--h', color='r', lw=1, markersize='4', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax3.legend(prop=font)

    ax4 = plt.subplot(234)
    ax4.plot(y, SVM, '--v', color='c', lw=1, markersize='4', label='SVM-RBF')
    ax4.plot(y, test_y, '--h', color='r', lw=1, markersize='4', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax4.legend(prop=font)

    ax5 = plt.subplot(235)
    ax5.plot(y, KNN, '--d', color='purple', lw=1, markersize='4', label='KNN')
    ax5.plot(y, test_y, '--h', color='r', lw=1, markersize='4', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax5.legend(prop=font)

    ax6 = plt.subplot(236)
    ax6.plot(y, RF, '--^', color='orange', lw=1, markersize='4', label='RF')
    ax6.plot(y, test_y, '--h', color='r', lw=1, markersize='4', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    ax6.legend(prop=font)

    fig1.tight_layout()
    plt.show()
    fig1.savefig('常规对比实验.svg', format='svg', dpi=250)
