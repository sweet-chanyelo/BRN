"""

"""
from sklearn.model_selection import KFold
from scipy.optimize import leastsq
import pandas as pd
import numpy as np
import math


def Inference(x1, x2, y, omega):
    sum_d = len(x1)  # 结论等级个数 =4
    sum_l = 2        # 规则数目=2

    belta = [x1, x2]  # 定义置信分布
    m = np.ones((sum_l, sum_d))  # 定义基本概率质量(2,4)
    for k in range(sum_l):
            m[k] = omega[k] * belta[k]
    # 转化为基本概率质量
    m_d = np.ones(sum_l)
    m_hat = np.ones(sum_l)
    m_wave = np.ones(sum_l)
    belta_sum = np.zeros(sum_l)
    for k in range(sum_l):
        m_hat[k] = 1 - np.sum(m[k])
        for i in range(sum_d):
            belta_sum[k] = np.sum(belta[k])
        m_d[k] = 1 - omega[k] * belta_sum[k]
        m_wave[k] = omega[k] * (1 - belta_sum[k])
    # print(m)  # 输出基本概率质量矩阵
    Mass = np.zeros(sum_d)
    # 定义输出置信度
    belief = np.zeros(sum_d)
    # &&&&&&&&&&&&&&&&&&&&&&&&&&规则融合&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
    # 总质量KI(k+1)
    conf = 0
    # 总质量
    for i in range(0, sum_d):
        for j in range(0, sum_d):
            if j != i:
                conf = conf + (m[0][i] * m[1][j])
            elif j == i:
                conf = conf
    coeff = 1 - conf - 1e-10
    # 规则融合
    for i in range(0, sum_d):
        Mass[i] = (m[0][i] * m[1][i] + m[0][i] * m_d[1] + m_d[0] * m[1][i]) / coeff
    m_wave_i = (m_wave[0] * m_wave[1] + m_wave[0] * m_hat[1] + m_hat[0] * m_wave[1]) / coeff
    m_hat_i = (m_hat[0] * m_hat[1]) / coeff
    m_d_i = m_hat_i + m_wave_i
    # 基本概率质量转化为置信度
    for d in range(0, sum_d):
        belief[d] = Mass[d] / (1 - m_hat_i)

    return belief


def CMAES(input1, input2, output, x, consequence):
    raw, col = input1.shape
    ouput_ture = np.zeros((raw, 1))
    # -----------------------------cmaes---------------------------#
    N = len(x)  # 维数
    xmean = x  # 初始值
    sigma = 0.1  # 步长(标准差)
    stopeval = 20  # 最大迭代次数   # stopfitness = 1e-10    # 截止误差
    # 选择策略参数设置
    lamda = 20 + math.floor(3 * np.log(N))  # 后代的数量,种群大小
    mu = lamda / 2  # 父代的数量
    weights = np.log(mu + 1 / 2) - np.log(range(1, int(mu)))  # 用于重组的父代矩阵?
    # print(weights, np.log(range(1, int(mu))))
    mu = math.floor(mu)  # 减一
    weights = weights / sum(weights)  # 重组权重矩阵归一化处理
    mueff = sum(weights) ** 2 / sum(weights ** 2)  # 方差的有效性
    # print(mueff)
    # 自适应策略参数设置
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # C阵的时间累积常数
    cs = (mueff + 2) / (N + mueff + 5)  # 步长控制的时间累积常数
    c1 = 2 / ((N + 1.3) ** 2 + mueff)  # 学习率 for rank - one update of C
    # c1 = 0.1
    # print(c1)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))  # and for rank - mu update
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs  # 抑制步长，一般选择1
    # 初始化策略参数与常数
    pc = np.zeros((N, 1))  # C阵的进化路径
    ps = np.zeros((N, 1))  # 步长的进化路径
    B = np.eye(N, N)  # B 定义了坐标系
    D = np.ones((N, 1))  # D 定义了规模
    C = B * np.diag(D ** 2) * B.T  # 协方差矩阵 C
    invsqrtC = B * np.diag(D ** -1) * B.T  # C ** （-1/2）
    eigeneval = 0  # B和D的跟踪更新
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))  # 期望 | | N(0, I) | | == norm(randn(N, 1))
    # -------------------- 一代循环 - -------------------------------
    counteval = 0  # 迭代次数
    arx = np.zeros((N, lamda))  # 定义最优参数
    arfitness = np.zeros(lamda)  # 定义局部最优适应度
    bestf = 0  # 定义全局最优适应度
    bestx = []  # 定义全局最优解
    while counteval < stopeval:  # 生成并评估所有后代, 第一代
        for k in range(0, lamda):  # 第一个后代
            vert = sigma * np.dot(B, np.multiply(D, np.random.rand(N, 1)))
            # print((np.dot(B, np.multiply(D, np.random.rand(N, 1)))))
            for kk in range(0, N):  # 设置上下界，参数约束
                arx[kk][k] = xmean[kk] + (vert[kk][0])  # 均值+步长*正态分布C
            # % % % % % % % % % % % % % % % % % % 投影算子加约束 % % % % % % % % % % % % % %
            for i in range(0, N):  # omega相对权重
                if arx[i][k] > 1:
                    arx[i][k] = 1
                if arx[i][k] < 0:
                    arx[i][k] = 0
            # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            belief = np.zeros((raw, col))
            for j in range(0, len(input1)):
                belief[j] = Inference(input1[j], input2[j], output[j], arx[:, k])  # 调用目标函数
                ouput_ture[j] = np.dot(belief[j], consequence)
            arfitness[k] = np.sum((ouput_ture - output) ** 2) / raw
        counteval = counteval + 1
        # 适应度排序并更新均值xmean
        arindex = np.argsort(arfitness)  # 最小化适应度及其索引
        arfitness = np.sort(arfitness)
        # print(arfitness, arindex)
        xold = xmean  # 保存旧的均值
        var = arindex[0:mu - 1]
        # print(var.shape, weights.shape)
        xmean = np.dot(arx[:, var], weights)  # 更新均值
        #  Cumulation: 更新进化路径
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma  # 共轭进化路径，指数平滑
        hsig = sum(ps ** 2) / (1 - (1 - cs) ** (2 * counteval / lamda)) / N < 2 + 4 / (N + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        # 自适应协方差矩阵 C
        artmp = (1 / sigma) * (
                (arx[:, var]) - np.tile(np.mat(xold).reshape(N, 1), (1, mu - 1)))  # mu difference vectors
        C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * artmp * np.diag(weights) * artmp.T  # 考虑旧的矩阵
        #  调整步长
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        # if sigma < 10e-4:
        #     sigma = 0.5
        # 根据C阵更新B和D
        if counteval - eigeneval > lamda / (c1 + cmu) / N / 10:  # to achieve O(N ^ 2):
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T  # 执行对称
            # print(C.shape)
            [D, B] = np.linalg.eig(C)  # 特征分解，B为特征变量，D为特征值
            D = np.diag(np.sqrt(np.abs(D)))  # D contains standard deviations now
            # print(counteval, arx)
            invsqrtC = B * np.diag(np.diag(D) ** -1) * B.T
        # ============================最优解及最优适应度===========================
        if counteval == 1:  # 初始化最优解
            bestf = arfitness[0]
            bestx = arx[:, arindex[0]]
        elif arfitness[0] < bestf:  # 更新最优解
            bestf = arfitness[0]
            bestx = arx[:, arindex[0]]
    # print('****', bestx, bestf)
    return bestx


def Integration(t, Data):
    """
    """
    input1 = Data[0]
    input2 = Data[1]
    # 输出
    df2 = pd.read_excel('input2.xlsx', "output", header=None)
    row, col = df2.shape
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df2.iloc[i]
    # 初始化
    raw, num_d = input1.shape   # 数据量，结论等级个数
    # print("规则单元{}的输入参数：{}, {}".format(t, input1, input2))
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]

    lamda = range(0, int(len(output) * 0.8), 1)
    train_x1 = input1[lamda]  # 训练集
    train_x2 = input2[lamda]  # 训练集
    train_y = output[lamda]  # 训练集标签
    test_x1 = np.delete(input1, lamda, 0)  # 测试集
    test_x2 = np.delete(input2, lamda, 0)  # 测试集
    test_y = np.delete(output, lamda, 0)  # 测试集标签

    # lamda = range(0, len(output), 5)
    # train_x1 = np.delete(input1, lamda, 0)  # 训练集
    # train_x2 = np.delete(input2, lamda, 0)  # 训练集
    # train_y = np.delete(output, lamda, 0)   # 训练集标签
    # test_x1 = input1[lamda]         # 测试集
    # test_x2 = input2[lamda]         # 测试集
    # test_y = output[lamda]          # 测试集标签
    # ￥￥￥￥￥￥￥￥￥￥￥￥￥运行￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥#
    belief = np.zeros((raw, num_d))
    omega = np.random.rand(2)
    # 优化训练
    # omega = CMAES(input1, input2, output, omega, consequence)
    # print('第{}个规则单元的结构参数为：{}'.format(t, para))

    # 测试集
    # Db = np.zeros((len(test_x1), num_d))
    # output_ture = np.zeros(len(test_x1))
    # for i in range(len(test_x1)):
    #         Db[i] = Inference(test_x1[i], test_x2[i], test_y[i], omega)  # 调用目标函数
    #         # 输出效用
    #         output_ture[i] = np.dot(Db[i], consequence)
    # # print('输出为:', ouput_ture)
    #
    # temp = 0
    # for i in range(len(test_y)):
    #     if abs(output_ture[i] - test_y[i]) > 0.1:
    #         temp += 1
    # error = temp / len(test_y)
    # error = np.sum((output_ture - test_y) ** 2) / len(test_x1)

    # 训练集
    Db = np.zeros((len(train_x1), num_d))
    ouput_ture = np.zeros(len(train_x1))
    for i in range(len(train_x1)):
        Db[i] = Inference(train_x1[i], train_x2[i], train_y[i], omega)  # 调用目标函数
        # 输出效用
        ouput_ture[i] = np.dot(Db[i], consequence)
    # print('输出为:', ouput_ture)

    # temp = 0
    # for i in range(len(train_y)):
    #     if abs(ouput_ture[i] - train_y[i]) > 0.1:
    #         temp += 1
    # error = temp / len(train_y)

    error = np.sum((ouput_ture - train_y) ** 2) / len(train_y)
    # error = (np.sum((ouput_ture - train_y) ** 2) / len(train_y)) ** 0.5
    # print('第{}个规则单元的测试误差为：{}'.format(t, error))
    for i in range(raw):
            belief[i] = Inference(input1[i], input2[i], output[i], omega)  # 调用目标函数

    res = [belief, error]
    return res