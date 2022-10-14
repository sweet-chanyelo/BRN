"""
EXPERT输出置信分布
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_excel('EXPERT输出置信分布.xlsx', 'data', header=None)
    raw, col = df1.shape
    expert = np.zeros((raw - 1, col))
    for i in range(1, raw):
        expert[i - 1] = df1.iloc[i]

    fig1 = plt.figure(figsize=(9, 8))
    x = np.arange(216)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置图例字体
    font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'bold', 'size': 12}  # 设置坐标轴字体
    plt.subplot(411)
    plt.bar(x, expert[:, 0])
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("Belief degree of N", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')
    #
    plt.subplot(412)
    plt.bar(x, expert[:, 1], color='b')
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("Belief degree of L", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')

    plt.subplot(413)
    plt.bar(x, expert[:, 2], color='navy')
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("Belief degree of M", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')

    plt.subplot(414)
    plt.bar(x, expert[:, 3], color='purple')
    plt.xlabel("Time of relay actions", font1)
    plt.ylabel("Belief degree of H", font1)
    plt.xticks(family='Times New Roman')
    plt.yticks(family='Times New Roman')

    fig1.tight_layout()
    plt.show()

    fig1.savefig('expert输出置信分布.svg', format='svg')