import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    df1 = pd.read_excel('input2.xlsx', 'data')
    row, col = df1.shape
    input = np.zeros((row - 1, col))
    for i in range(1, row):
        input[i - 1] = df1.iloc[i]
    print(max(input[:, 5]), min(input[:, 5]))

    df2 = pd.read_excel('input2.xlsx', 'output')
    # print(df2)
    row, col = df2.shape
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iat[i, 0]
    print(len(output))
    x = np.linspace(0, len(output), len(output))

    fig2 = plt.figure(figsize=(9, 6))
    index = [8, 12, 25, 82, 103, 105, 108, 115, 119, 126, 129, 131, 172, 173, 180, 191, 212]
    test_y = output[index]
    plt.plot(test_y, '-.*', color='g', lw=1, markersize='2.8')

    fig1 = plt.figure(figsize=(9, 6))
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 12}  # 设置坐标轴字体

    plt.subplot(231)
    plt.plot(input[:, 1], '-.*', color='g', lw=1, markersize='2.8')
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("SIT(ms)", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')

    plt.subplot(232)
    plt.plot(input[:, 3], '-.v', color='g', lw=1, markersize='2.8')
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("RW($ \Omega $)", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')

    plt.subplot(233)
    plt.plot(input[:, 5], '-.o', color='g', lw=1, markersize='2.8')
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("RV(V)", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')

    plt.subplot(212)
    plt.plot(output, '-.', color='r', lw=1)
    plt.scatter(x, output, color='', marker='p', edgecolors='b', s=24)
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("The real results", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    # plt.grid(axis='y', linestyle='-.')

    fig1.tight_layout()
    plt.show()

    # fig1.savefig('输入与输出.svg', format='svg')