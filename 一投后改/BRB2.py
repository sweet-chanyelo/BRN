"""
BRB2,(1,2)
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math


def BRB(input, output, para, ref):
    lamda = len(input)
    num_t = 2
    sum_t = [4, 4]
    num_l = 16
    num_d = 4

    delta = para[0: num_t]
    theta = para[num_t: num_t + num_l]
    belta = np.array(para[num_t + num_l:]).reshape((num_l, num_d))
    # print(para, belta)
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&输入匹配度&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
    belief = np.zeros((lamda, num_d))
    ouput_ture = np.zeros(lamda)
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
        for i in range(num_l):
            omega[i] = (theta[i] * alpha_k[i] / sum(theta * alpha_k))
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

    err = np.sum((ouput_ture - output) ** 2) / lamda
    # print(err)
    return err, ouput_ture


def CMAES(input, output, x, Ax):
    # lamda = len(input)
    num_t = 2
    num_l = 9
    num_d = 4
    # -----------------------------cmaes---------------------------#
    N = len(x)             # 维数
    xmean = x              # 初始值
    sigma = 0.1            # 步长(标准差)
    stopeval = 10          # 最大迭代次数
    # stopfitness = 10       # 截止误差
    # 选择策略参数设置
    lamda = 20 + math.floor(3 * np.log(N))        # 后代的数量,种群大小
    mu = lamda / 2                                # 父代的数量
    weights = np.log(mu + 1 / 2) - np.log(range(1, int(mu)))  # 用于重组的父代矩阵
    # print(weights, np.log(range(1, int(mu))))
    mu = math.floor(mu)                            # 减一
    weights = weights / sum(weights)               # 重组权重矩阵归一化处理
    mueff = sum(weights) ** 2 / sum(weights ** 2)  # 方差的有效性
    # print(mueff)
    # 自适应策略参数设置
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # C阵的时间累积常数
    cs = (mueff + 2) / (N + mueff + 5)              # 步长控制的时间累积常数
    c1 = 2 / ((N + 1.3) ** 2 + mueff)               # 学习率 for rank - one update of C
    # c1 = 0.1
    # print(c1)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))  # and for rank - mu update
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs  # 抑制步长，一般选择1
    # 初始化策略参数与常数
    pc = np.zeros((N, 1))                # C阵的进化路径
    ps = np.zeros((N, 1))                # 步长的进化路径
    B = np.eye(N, N)                     # B 定义了坐标系
    D = np.ones((N, 1))                  # D 定义了规模
    C = B * np.diag(D ** 2) * B.T      # 协方差矩阵 C
    invsqrtC = B * np.diag(D ** -1) * B.T  # C ** （-1/2）
    eigeneval = 0                        # B和D的跟踪更新
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))  # 期望 | | N(0, I) | | == norm(randn(N, 1))
    # out_dat = []
    # out_datx = []  # 用于画输出图
    # -------------------- 一代循环 - -------------------------------
    counteval = 0  # 迭代次数
    arx = np.zeros((N, lamda))       # 定义最优参数
    arfitness = np.zeros(lamda)  # 定义局部最优适应度
    bestf = 0     # 定义全局最优适应度
    bestx = []    # 定义全局最优解                    arx[i][k] = 1

    while counteval < stopeval:  # 生成并评估所有后代, 第一代
        for k in range(0, lamda):  # 第一个后代
            vert = sigma * np.dot(B, np.multiply(D, np.random.rand(N, 1)))
            # print(vert)
            for kk in range(0, N):  # 设置上下界，参数约束
                arx[kk][k] = xmean[kk] + abs(vert[kk][0])   # 均值+步长*正态分布C
            # print(arx[:, k])
    # % % % % % % % % % % % % % % % % % % 投影算子加约束 % % % % % % % % % % % % % %
            for i in range(num_t):  # delta前提属性
                arx[i] = arx[i] / np.max(arx[0:num_t])
            for i in range(num_l):  # theta规则权重
                arx[num_t + i] = arx[num_t + i] / np.max(arx[num_t:num_t + num_l])
            for i in range(0, num_l):  # belta相对权重
                arx[((num_l + num_t) + i * num_d): ((num_l + num_t) + i * num_d) + num_d, k] \
                    = arx[((num_l + num_t) + i * num_d): ((num_l + num_t) + i * num_d) + num_d, k] / \
                      np.sum(arx[((num_l + num_t) + i * num_d): ((num_l + num_t) + i * num_d) + num_d, k])
                # print(arx[(11 + i * sum_d): (11 + i * sum_d) + sum_d, k])
            # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            arfitness[k], blank = BRB(input, output, arx[:, k], Ax)  # 调用目标函数
        counteval = counteval + 1
        # 适应度排序并更新均值xmean
        arindex = np.argsort(arfitness)  # 最小化适应度及其索引
        arfitness = np.sort(arfitness)
        # print(arfitness, arindex)
        xold = xmean                  # 保存旧的均值
        var = arindex[0:mu-1]
        # print(var.shape, weights.shape)
        xmean = np.dot(arx[:, var], weights)  # 更新均值
        #  Cumulation: 更新进化路径
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma  # 共轭进化路径，指数平滑
        hsig = sum(ps ** 2) / (1 - (1 - cs) ** (2 * counteval / lamda)) / N < 2 + 4 / (N + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        # 自适应协方差矩阵 C
        artmp = (1 / sigma) * ((arx[:, var]) - np.tile(np.mat(xold).reshape(N, 1), (1, mu-1)))  # mu difference vectors
        C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C)\
            + cmu * artmp * np.diag(weights) * artmp.T  # 考虑旧的矩阵
        #  调整步长
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        # if sigma < 10e-4:
        #     sigma = 0.5
        # 根据C阵更新B和D
        if counteval - eigeneval > lamda / (c1 + cmu) / N / 10:  # to achieve O(N ^ 2):
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T   # 执行对称
            # print(C.shape)
            [D, B] = np.linalg.eig(C)          # 特征分解，B为特征变量，D为特征值
            D = np.diag(np.sqrt(np.abs(D)))          # D contains standard deviations now
            # print(counteval, arx)
            invsqrtC = B * np.diag(np.diag(D) ** -1) * B.T
    # ============================最优解及最优适应度===========================
        if counteval == 1:           # 初始化最优解
            bestf = arfitness[0]
            bestx = arx[:, arindex[0]]
        elif arfitness[0] < bestf:   # 更新最优解
            bestf = arfitness[0]
            bestx = arx[:, arindex[0]]
            FES = counteval
        # print('第{}代的误差为{}'.format(counteval, arfitness[0]))
    # print('最小误差为：{}'.format(bestf))
    return bestx


if __name__ == '__main__':
    # ***********************************导入数据***************************************#
    df1 = pd.read_excel('input2.xlsx', "data", header=None)
    row, col = df1.shape
    input = np.zeros((row-1, 2))
    for i in range(1, row):
        input[i-1][0] = df1.iat[i, 1]
        input[i-1][1] = df1.iat[i, 2]
    print(max(input[:, 1]), min(input[:, 1]))
    # 输出
    df2 = pd.read_excel('input2.xlsx', "output", header=None)
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iloc[i]
    # print(output)

    # *****************************参数设置*************************************#
    num_t = 2   # 第j个BRB的前提属性
    num_l = 16  # 规则数目
    num_d = 4   # 评价等级

    feature3 = [1.98, 2.01, 2.04, 2.084]
    feature4 = [14, 14.7, 15.4, 16.3]

    Ax = [feature3, feature4]      # 参考值集
    delta = [1, 1]                 # 属性权重
    # theta = np.random.rand(num_l)  # 规则权重

    theta = np.zeros(num_l)  # 定义规则权重
    df4 = pd.read_excel('置信度.xlsx', "theta", header=None)
    for i in range(0, num_l):
        theta[i] = df4.iat[i + 1, 0]
    # belta = np.random.rand(num_l, num_d)  # 置信度
    belta = np.zeros((num_l, num_d))  # 定义置信度
    df3 = pd.read_excel('置信度.xlsx', "belta", header=None)
    for i in range(0, num_l):
        belta[i] = df3.iloc[i + 1]

    for i in range(num_l):  # 归一化
        belta[i] = belta[i] / sum(belta[i])
    # print(belta)
    belta_list = belta.flatten()
    para = np.append(delta, theta)
    para = list(para) + list(belta_list)
    # print(para)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^BRB^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    lamda = range(0, len(output), 5)
    train_x = np.delete(input, lamda, 0)
    train_y = np.delete(output, lamda, 0)
    test_x = input[lamda]
    test_y = output[lamda]
    # print(len(lamda))
    print('###############开始优化####################333')
    para = CMAES(train_x, train_y, para, Ax)
    # para = CMAES(input, output, para, Ax)
    print('###############优化结束####################333')
    print('最优参数为：', para)

    err, ouput_ture = BRB(input, output, para, Ax)
    print('测试误差为：{}'.format(err))
    # 画图
    plt.plot(ouput_ture)
    plt.plot(output)
    plt.show()

    # 输出
    # error, belief = BRB(input, output, para, Ax)
    # print('拟合误差为：{}'.format(error))
    # output_excel = pd.DataFrame(belief)
    # writer = pd.ExcelWriter('BRB2.xlsx')
    # output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    # writer.save()