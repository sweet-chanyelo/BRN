"""

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_excel('误差MSE.xlsx', 'data', header=None)
    raw, col = df1.shape
    mse = np.zeros(raw - 1)
    for i in range(1, raw):
        mse[i - 1] = df1.iloc[i]

    df2 = pd.read_excel('输出方差EV.xlsx', "data", header=None)
    row, col = df2.shape
    ev = np.zeros(row - 1)
    for i in range(1, row):
        ev[i - 1] = df2.iloc[i]

    fig1 = plt.figure(1)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 12}  # 设置坐标轴字体

    ax1 = plt.subplot(111)
    ax1.plot(mse, '--s', color='b', lw=1, markersize='2.8', label='The MSE of BRN')
    plt.xlabel("Number of test", font1)
    # plt.ylabel("MSE", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    plt.legend(loc='upper right', prop=font)

    ax11 = ax1.twinx()
    ax11.plot(ev, '--v', color='r', lw=1, markersize='2.8', label='The EV of BRN')
    plt.yticks(family='Times New Roman')
    plt.yscale('log')  # 科学计数法
    # plt.ylabel("EV", font1)
    # plt.grid(axis='y', linestyle='-.')
    plt.legend(loc='upper right', bbox_to_anchor=(0.975, 0.92), prop=font)
    plt.show()

    fig1.savefig('MSE与EV.svg', format='svg')