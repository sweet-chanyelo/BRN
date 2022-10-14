"""
BRB19,(7,8)
优化方法：PSO，精度0.082
规则数：9
参数：47
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math
from sko.PSO import PSO
from sko.GA import GA
import time


def brb(p, input):
    # 结构参数初始化
    num_t = 2  # 输入个数
    sum_t = [3, 3]  # 参考值个数
    num_l = 9  # 规则数
    num_d = 4  # 输出等级个数

    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    feature1 = [0.38, 0.4, 0.43]
    feature2 = [0.41, 1, 1.45]
    ref = [feature1, feature2]  # 参考值集

    # 模型参数初始化
    delta = p[0: num_t]
    theta = p[num_t: num_t + num_l]
    belta = np.array(p[num_t + num_l:]).reshape((num_l, num_d))
    # 约束
    for i in range(0, num_l):  # belta相对权重
        if np.sum(belta[i]) > 1:
            belta[i, :] = belta[i, :] / np.sum(belta[i])
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&输入匹配度&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

    lamda = len(input)  # 训练集长度
    belief = np.zeros((lamda, num_d))
    output_ture = np.zeros(lamda)
    for l in range(0, lamda):
        # 计算输入匹配度
        # alpha中保存每个前提属性相对于各自的参考值的匹配度，
        alpha = [[], []]
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
        # 按照排列组合建立规则，确定最终的规则匹配度矩阵（L乘M）
        rule = list(itertools.product(alpha[0], alpha[1]))
        # ++++++++++++++++++++++++++激活权重+++++++++++++++++++++++++++++++++++#
        alpha_k = np.ones(num_l)
        omega = np.zeros(num_l)
        # 生成激活权重
        for k in range(num_l):
            for i in range(num_t):
                alpha_k[k] = alpha_k[k] * (rule[k][i] ** float(delta[i]))
        # if np.sum(theta * alpha_k) != 0:
        for i in range(num_l):
            omega[i] = (theta[i] * alpha_k[i] / np.sum(theta * alpha_k))
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
        output_ture[l] = np.dot(belief[l], consequence)
    return output_ture, belief


def demo_func(p):
    # print(p)
    # ***********************************导入数据***************************************#
    df1 = pd.read_excel('input2.xlsx', "data", header=None)
    row, col = df1.shape
    input = np.zeros((row - 1, 2))
    for i in range(1, row):
        input[i - 1][0] = df1.iat[i, 7]
        input[i - 1][1] = df1.iat[i, 8]
    # print(max(input[:, 1]), min(input[:, 1]))
    # 输出
    df2 = pd.read_excel('input2.xlsx', "output", header=None)
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iloc[i]
    # print(output)
    # 数据划分
    lamda = range(0, int(len(output) * 0.8), 1)
    train_x = input[lamda]  # 训练集
    train_y = output[lamda]  # 训练集标签
    test_x = np.delete(input, lamda, 0)  # 测试集
    test_y = np.delete(output, lamda, 0)  # 测试集标签
    # index = range(0, len(output), 5)
    # train_x = np.delete(input, index, 0)  # 训练集
    # train_y = np.delete(output, index, 0)
    # test_x = input[index]  # 测试集
    # test_y = output[index]

    output_ture = np.zeros(len(train_y))
    output_ture, belief = brb(p, train_x)

    err = np.sum((output_ture - train_y) ** 2) / len(train_y)
    # print(err)
    return err


if __name__ == '__main__':
    start = time.perf_counter()  # 开始时间
    N = 47
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
    input = np.zeros((row - 1, 2))
    for i in range(1, row):
        input[i - 1][0] = df1.iat[i, 5]
        input[i - 1][1] = df1.iat[i, 8]
    # print(max(input[:, 1]), min(input[:, 1]))
    # 输出
    df2 = pd.read_excel('input2.xlsx', "output", header=None)
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iloc[i]

    # 数据划分
    lamda = range(0, int(len(output) * 0.8), 1)
    train_x = input[lamda]  # 训练集
    train_y = output[lamda]  # 训练集标签
    test_x = np.delete(input, lamda, 0)  # 测试集
    test_y = np.delete(output, lamda, 0)  # 测试集标签
    # index = range(0, len(output), 5)
    # train_x = np.delete(input, index, 0)  # 训练集
    # train_y = np.delete(output, index, 0)
    # test_x = input[index]  # 测试集
    # test_y = output[index]

    blank, belief = brb(best_x, input)
    # 输出
    output_excel = pd.DataFrame(belief)
    writer = pd.ExcelWriter('BRB19.xlsx')
    output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    writer.save()