"""
作者——张春潮
时间——2020/8/19
主题——继电器故障状态评估
    1）采用3个特征量
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import itertools
import time
import math
from sko.PSO import PSO


def brb(p, input):
    # 结构参数初始化
    num_t = 3  # 输入个数
    sum_t = [5, 5, 5]  # 参考值个数
    num_l = 125  # 规则数
    num_d = 4  # 输出等级个数

    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    # 第一个前提属性参考值(由用户给定)
    feature1 = [min(input[:, 0]), min(input[:, 0]) / 2, 0, max(input[:, 0]) / 2, max(input[:, 0])]
    # 第二个前提属性参考值(由用户给定)
    feature2 = [min(input[:, 1]), min(input[:, 1]) / 2, 0, max(input[:, 1]) / 2, max(input[:, 1])]
    # 第三个前提属性参考值(由用户给定)
    feature3 = [min(input[:, 2]), min(input[:, 2]) / 2, 0, max(input[:, 2]) / 2, max(input[:, 2])]

    ref = [feature1, feature2, feature3]  # 参考值集

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
        alpha = [[], [], []]
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
        rule = list(itertools.product(alpha[0], alpha[1], alpha[2]))
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
    # 导入数据
    df = pd.read_excel('input2.xlsx', sheet_name='data', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        input[i - 1] = df.iloc[i]

    # 导入标签数据
    df2 = pd.read_excel('input2.xlsx', sheet_name='output', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iloc[i]
    # print(output)

    # 主成分分析
    pca = PCA(3)  # 保留所有成分
    pca.fit(input)
    low_d = pca.transform(input)  # 降低维度

    output_ture, belief = brb(p, low_d)

    err = (np.sum((output_ture - output) ** 2) / len(output)) ** 0.5
    # print(err)
    return err


if __name__ == '__main__':
    start = time.perf_counter()  # 开始时间
    # 优化
    N = 628
    pso = PSO(func=demo_func, dim=N, pop=40, max_iter=100, lb=np.zeros(N), ub=np.ones(N), w=0.8, c1=0.5, c2=0.5)
    best_x, best_y = pso.run()

    # 输出
    end = time.perf_counter()  # 停止时间
    print('运行时间：%s' % (end - start))
    print('best x is', best_x)
    print('best y is', best_y)

    # 导入数据
    df = pd.read_excel('input2.xlsx', sheet_name='data', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        input[i - 1] = df.iloc[i]

    # 导入标签数据
    df2 = pd.read_excel('input2.xlsx', sheet_name='output', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iloc[i]
    # print(output)

    # 主成分分析
    pca = PCA(3)  # 保留所有成分
    pca.fit(input)
    low_d = pca.transform(input)  # 降低维度
    print("------------------------------主成分分析---------------------------")
    print("特征向量", pca.explained_variance_)  # 返回特征向量
    print("贡献率", pca.explained_variance_ratio_)  # 返回各个成分各自的方差百分比(也称贡献率）
    # print(low_d)

    # 数据划分
    lamda = range(0, len(output), 10)
    dataset = np.delete(input, lamda, 0)  # 训练集
    labelset = np.delete(output, lamda, 0)  # 训练集标签
    test_dataset = input[lamda]  # 测试集
    test_labelset = output[lamda]  # 测试集标签

    blank, belief = brb(best_x, test_dataset)
    # 输出
    # output_excel = pd.DataFrame(belief)
    # writer = pd.ExcelWriter('PCA.xlsx')
    # output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    # writer.save()