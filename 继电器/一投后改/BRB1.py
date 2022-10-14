"""
BRB1,(1)
优化方法：PSO，精度0.053
规则数：4
参数：21
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math
from sko.PSO import PSO
import time


def brb(para, input):
    lamda = len(input)
    num_t = 1
    sum_t = [4]
    num_l = 4
    num_d = 4

    feature7 = [1.98, 2.01, 2.04, 2.084]
    ref = [feature7]  # 参考值集

    delta = para[0: num_t]
    theta = para[num_t: num_t + num_l]
    belta = np.array(para[num_t + num_l:]).reshape((num_l, num_d))
    for i in range(num_l):
        if sum(belta[i]) >= 1:
            belta[i] = belta[i] / sum(belta[i])
    # print(para, belta)
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&输入匹配度&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
    belief = np.zeros((lamda, num_d))
    ouput_ture = np.zeros(lamda)
    for l in range(0, lamda):
        # 计算输入匹配度
        # alpha中保存每个前提属性相对于各自的参考值的匹配度，
        alpha = [[]]
        for i in range(0, num_t):
            # 前提属性的参考值个数
            r = int(sum_t[i])
            alpha_i = np.zeros(r)
            for j in range(0, r):
                if j + 1 != r and ref[i][j] <= input[l][i] < ref[i][j + 1]:
                    alpha_i[j] = (ref[i][j + 1] - input[l][i]) / (ref[i][j + 1] - ref[i][j])
                    alpha_i[j + 1] = (input[l][i] - ref[i][j]) / (ref[i][j + 1] - ref[i][j])
                elif j + 1 == r and ref[i][j] <= input[l][i]:
                    alpha_i[j] = 1
                elif j + 1 == r and ref[i][0] >= input[l][i]:
                    alpha_i[0] = 1
            alpha[i] = alpha_i
        # print("**", alpha)
        # ++++++++++++++++++++++++++激活权重+++++++++++++++++++++++++++++++++++#
        omega = np.zeros(num_l)
        # 生成激活权重

        for i in range(num_l):
            omega[i] = (theta[i] * alpha[0][i] / sum(theta * alpha[0]))
        # print('@@##', omega)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$规则融合基于ER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
        # 定义辅助变量
        a, b = np.ones(num_d), 1
        for i in range(num_d):
            for j in range(num_l):
                a[i] = a[i] * (omega[j] * belta[j][i] + 1 - omega[j] * sum(belta[j]))
                b = np.prod(1 - omega * sum(belta[j]))
        mu = 1 / (sum(a) - (num_d - 1) * b)
        for i in range(num_d):
            belief[l][i] = mu * (a[i] - b) / (1 - mu * np.prod(1 - omega))
        # print(belief)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@输出@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
        # 输出效用
        ouput_ture[l] = np.dot(belief[l], consequence)
    return ouput_ture, belief


def demo_func(p):
    # print(p)
    # ***********************************导入数据***************************************#
    df1 = pd.read_excel('input2.xlsx', "data", header=None)
    row, col = df1.shape
    input = np.zeros((row - 1, 1))
    for i in range(1, row):
        input[i - 1][0] = df1.iat[i, 1]
    # print(max(input[:, 1]), min(input[:, 1]))
    # 输出
    df2 = pd.read_excel('input2.xlsx', "output", header=None)
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iloc[i]
    # print(output)

    index = range(0, len(output), 5)
    train_x = np.delete(input, index, 0)  # 训练集
    train_y = np.delete(output, index, 0)
    test_x = input[index]  # 测试集
    test_y = output[index]

    output_ture, belief = brb(p, train_x)
    err = np.sum((output_ture - train_y) ** 2) / len(train_y)
    # print(err)
    return err


if __name__ == '__main__':
    start = time.perf_counter()  # 开始时间
    N = 21
    pso = PSO(func=demo_func, dim=N, pop=40, max_iter=150, lb=np.zeros(N), ub=np.ones(N), w=0.8, c1=0.5, c2=0.5)
    best_x, best_y = pso.run()

    # 输出
    end = time.perf_counter()  # 停止时间
    print('运行时间：%s' % (end - start))
    print('best x is', best_x)
    print('best y is', best_y)
# -----------------运行-----------------------------------------------------------------------------------%#
    # ***********************************导入数据***************************************#
    df1 = pd.read_excel('input2.xlsx', "data", header=None)
    row, col = df1.shape
    input = np.zeros((row-1, 1))
    for i in range(1, row):
        input[i-1][0] = df1.iat[i, 1]
    print(max(input[:, 0]), min(input[:, 0]))
    # 输出
    df2 = pd.read_excel('input2.xlsx', "output", header=None)
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iloc[i]
    # print(output)

    # *****************************参数设置*************************************#
    num_t = 1   # 第j个BRB的前提属性
    num_l = 4  # 规则数目
    num_d = 4   # 评价等级

    lamda = range(0, len(output), 5)
    train_x = np.delete(input, lamda, 0)
    train_y = np.delete(output, lamda, 0)
    test_x = input[lamda]
    test_y = output[lamda]
    blank, belief = brb(best_x, input)
    # 输出
    # output_excel = pd.DataFrame(belief)
    # writer = pd.ExcelWriter('BRB1.xlsx')
    # output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    # writer.save()