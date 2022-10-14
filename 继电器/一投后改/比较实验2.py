"""
BRN>>
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_excel('BRN15组输出.xlsx', 'data', header=None)
    df2 = pd.read_excel('BRB15组输出.xlsx', 'data', header=None)
    df3 = pd.read_excel('FIE15组输出.xlsx', 'data', header=None)
    df4 = pd.read_excel('RIMER15组输出.xlsx', 'data', header=None)
    raw, col = df1.shape
    BRN = np.zeros(raw - 1)
    BRB = np.zeros(raw - 1)
    FIM = np.zeros(raw - 1)
    RIMER = np.zeros(raw - 1)
    for i in range(1, raw):
        BRN[i - 1] = df1.iloc[i]
        BRB[i - 1] = df2.iloc[i]
        FIM[i - 1] = df3.iloc[i]
        RIMER[i - 1] = df4.iloc[i]
    # df5 = pd.read_excel('测试输出.xlsx', 'data2', header=None)
    df6 = pd.read_excel('input2.xlsx', 'output', header=None)
    raw, col = df6.shape
    output = np.zeros(raw - 1)
    for i in range(1, raw):
        output[i - 1] = df6.iloc[i]
    # print(len(output))

    index = [8, 12, 25, 82, 103, 105, 108, 119, 129, 131, 172, 173, 180, 191, 212]
    train_y = np.delete(output, index, 0)
    test_y = output[index]

    # 画图
    y = np.linspace(1, 15, 15)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 14}  # 设置坐标轴字体
    fig1 = plt.figure(figsize=(10, 4.5))
    plt.rcParams['xtick.direction'] = 'in'  # X轴下标朝内
    plt.rcParams['ytick.direction'] = 'in'  # Y轴下标朝内

    plt.plot(y, BRN, '--s', color='b', lw=1.3, markersize='4.5', label='BRN')
    plt.plot(y, BRB, '-->', lw=1.3, markersize='5.5', label='BRB-C')
    plt.plot(y, FIM, '--*', color='g', lw=1.3, markersize='6', label='FIE-URC')
    plt.plot(y, RIMER, '--v', color='c', lw=1.3, markersize='5.5', label='RIMER-PCA')
    plt.plot(y, test_y, '--h', color='r', lw=1.3, markersize='4.5', label='The actual status')
    plt.xlabel("Number of test", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.ylabel("Evaluation result", font1)
    # plt.grid(axis='y', linestyle='-.')
    plt.legend(prop=font)

    plt.show()
    fig1.savefig('常规对比实验2.svg', format='svg', dpi=250)