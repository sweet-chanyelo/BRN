"""
作者——张春潮
时间——2020/8/19
主题——继电器故障状态评估
    1）采用两个特征量
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import itertools
import time
import math


def aggregation(m, omega, belta):
    sum_l = len(belta)     # 规则库数目
    sum_d = len(belta[0])  # d为结论等级个数
    pointer = []
    for i in range(0, sum_l):
        if omega[i] > 0:
            pointer.append(i)
    # 转化为基本概率质量
    m_d = np.ones(sum_l)
    m_hat = np.ones(sum_l)
    m_wave = np.ones(sum_l)
    belta_sum = np.zeros(sum_l)
    for k in range(sum_l):
        m_hat[k] = 1 - np.sum(m[k])
        for i in range(sum_d):
            belta_sum[k] = belta_sum[k] + belta[k][i]
        m_d[k] = 1 - omega[k] * belta_sum[k]
        m_wave[k] = omega[k] * (1 - belta_sum[k])
    # print(m)  # 输出基本概率质量矩阵
    # 定义迭代变量
    m_i = m[pointer[0]]
    m_d_i = m_d[pointer[0]]
    m_hat_i = m_hat[pointer[0]]
    m_wave_i = m_wave[pointer[0]]
    # 定义输出置信度
    belief = np.zeros(sum_d)
    # 规则融合
    for k in range(0, len(pointer)-1):
        # 总质量KI(k+1)
        conf = 0
        # 总质量
        for i in range(0, sum_d):
            for j in range(0, sum_d):
                if j != i:
                    conf = conf + (m_i[i] * m[pointer[k + 1]][j])
                elif j == i:
                    conf = conf
        coeff = 1 - conf
        # 规则融合
        for i in range(0, sum_d):
            m_i[i] = (m_i[i] * m[pointer[k + 1]][i] + m_i[i] * m_d[pointer[k + 1]] + m_d_i * m[pointer[k + 1]][i]) / coeff
        m_wave_i = (m_wave_i * m_wave[pointer[k + 1]] + m_wave_i * m_hat[pointer[k + 1]] + m_hat_i * m_wave[pointer[k + 1]]) / coeff
        m_hat_i = (m_hat_i * m_hat[pointer[k + 1]]) / coeff
        m_d_i = m_hat_i + m_wave_i
    # 输出最终基本概率质量
    # print(m_i)
    # print(m_d_i)
    # print(m_hat_i)
    # print(m_wave_i)
    # 基本概率质量转化为置信度
    for d in range(0, sum_d):
        belief[d] = m_i[d] / (1 - m_hat_i)
    # 输出最终置信度
    return belief


def brb(data, ref, p):
    test_u = data[0]  # 训练集
    output = data[1]  # 训练集标签
    eval_z = np.zeros(len(output))  # 定义模型估计值
    consequent = data[2]  # 评估等级
    T = len(test_u)
    # 样本维度既前提属性个数
    sum_m = len(ref)
    # 给定置信规则库的初始值,确定各参数
    theta = p[0:9]  # 规则权重
    delta = p[9:11]  # 属性权重
    belta = np.array(p[11:]).reshape((9, 3))  # 置信度
    # print(theta, delta, belta)  # 检验
    sum_l = len(belta)     # 规则库数目
    sum_d = len(belta[0])  # d为结论等级个数
    # 确定各前提属性的参考值个数
    sum_t = np.ones(sum_m)
    for j in range(0, sum_m):
        sum_t[j] = int(len(ref[j]))
    ture_z = np.zeros(T)       # 定义估计值
    err = 0
    z_wave = np.zeros(T)       # 定义输出误差
    omega = np.zeros((T, sum_l))  # 定义激活权重

    for l in range(0, T):      # 计算输入匹配度
        # alpha中保存每个前提属性相对于各自的参考值的匹配度，
        alpha = [[], []]
        for i in range(0, sum_m):
            # 前提属性的参考值个数
            r = int(sum_t[i])
            alpha_i = np.zeros(r)
            for j in range(0, r):
                if j + 1 != r and ref[i][j] <= test_u[l][i] < ref[i][j + 1]:
                    alpha_i[j] = (ref[i][j + 1] - test_u[l][i]) / (ref[i][j + 1] - ref[i][j])
                    alpha_i[j + 1] = (test_u[l][i] - ref[i][j]) / (ref[i][j + 1] - ref[i][j])
                elif j + 1 == r and ref[i][j] <= test_u[l][i]:
                    alpha_i[j] = 1
                elif j + 1 == r and ref[i][0] >= test_u[l][i]:
                    alpha_i[0] = 1
            alpha[i] = alpha_i

        # 按照排列组合建立规则，确定最终的规则匹配度矩阵（L乘M）
        rule = list(itertools.product(alpha[0], alpha[1]))
        # 检验
        # print(rule)
        # 由输入匹配度确立规则激活权重
        alpha_k = np.ones(sum_l)

        pointer = []
        # 生成激活权重
        for k in range(sum_l):
            for i in range(0, sum_m):
                alpha_k[k] = alpha_k[k] * (rule[k][i] ** float(delta[i]))
            if alpha_k[k] > 0:
                pointer.append(k)
        for i in range(len(pointer)):
            omega[l][pointer[i]] = (theta[pointer[i]] * alpha_k[pointer[i]] / sum(theta * alpha_k))
        # print(pointer)
    m = np.ones((sum_l, sum_d))
    for l in range(0, T):      # 输出方程
        # 规则的融合
        for k in range(sum_l):
            m[k] = omega[l][k] * belta[k]
        belief = aggregation(m, omega[l], belta)
        # print(belief)                           # 检验置信分布
        ture_z[l] = np.mat(belief) * np.mat(consequent).T   # 输出效用
        z_wave[l] = output[l] - ture_z[l]
        err = err + z_wave[l] ** 2  # 均方和

    print("误差为：%s" % (err / T))
    if len(output) < 100:
        plt.figure(2)
        plt.plot(ture_z)
    return (err / T)


def cmaes(data, ref, x):
    # ==================初始化==========================#
    sum_d = 3                   # 结论等级个数
    sum_m = len(ref)            # 属性个数
    sum_l = 9                   # 规则库数目
# -----------------------------cmaes---------------------------#
    FES = 0
    N = len(x)             # 维数
    xmean = x              # 初始值
    sigma = 0.5            # 步长(标准差)
    stopeval = 10        # 最大迭代次数   # stopfitness = 1e-10    # 截止误差
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
                arx[kk][k] = xmean[kk] + (vert[kk][0])   # 均值+步长*正态分布C
            # print(arx[:, k])
    # % % % % % % % % % % % % % % % % % % 投影算子加约束 % % % % % % % % % % % % % %
            for i in range(0, sum_l):  # theta相对权重
                if arx[i][k] < 0:
                    arx[i][k] = 0
                arx[i][k] = arx[i][k] / max(arx[: sum_l, k])
            for i in range(sum_l, sum_l + sum_m):  # delta相对权重
                if arx[i][k] > 1:
                    arx[i][k] = 1
                elif arx[i][k] < 0:
                    arx[i][k] = 0
                # arx[i][k] = arx[i][k] / max(arx[sum_l: (sum_l + sum_t), k])
            for i in range(0, sum_l):  # belta相对权重
                for j in range(0, sum_d):
                    if arx[((sum_l + sum_m) + i * sum_d) + j, k] < 0:
                        arx[((sum_l + sum_m) + i * sum_d) + j, k] = 0
                arx[((sum_l + sum_m) + i * sum_d): ((sum_l + sum_m) + i * sum_d) + sum_d, k] \
                    = arx[((sum_l + sum_m) + i * sum_d): ((sum_l + sum_m) + i * sum_d) + sum_d, k] / \
                    sum(arx[((sum_l + sum_m) + i * sum_d): ((sum_l + sum_m) + i * sum_d) + sum_d, k])
                # print(arx[(11 + i * sum_d): (11 + i * sum_d) + sum_d, k])
            # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            arfitness[k] = brb(data, ref, arx[:, k])  # 调用目标函数
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
    # print(FES, bestf)
    return bestx


if __name__ == '__main__':
    # 前提属性组成的一个样本(由用户给定，可由excel导入)
    df1 = pd.read_excel('data3.xls', sheet_name='input-183', header=None)
    N, dim = df1.shape                # 2008row，6col
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df1.iat[i, j]
    # 导入标签数据
    df2 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
    N, dim = df2.shape  # 2008row，1col
    output = np.zeros(N - 1)  # 默认为只有1列（单输出）
    for i in range(1, N):
        output[i - 1] = df2.iat[i, 0]
    # print(input)  # 检验

    # 主成分分析
    pca = PCA(2)  # 保留所有成分
    pca.fit(input)
    low_d = pca.transform(input)  # 降低维度
    writer = pd.ExcelWriter('pca.xls')
    pd.DataFrame(low_d).to_excel(writer, sheet_name='pca', startcol=0, index=False)  # 保存结果
    print("------------------------------主成分分析---------------------------")
    print("特征向量", pca.explained_variance_)        # 返回特征向量
    print("贡献率", pca.explained_variance_ratio_)  # 返回各个成分各自的方差百分比(也称贡献率）
    # print(low_d)

    # 第一个前提属性参考值(由用户给定)
    R1 = [min(low_d[:, 0]), 0, max(low_d[:, 0])]
    # 第二个前提属性参考值(由用户给定)
    R2 = [min(low_d[:, 1]), 0, max(low_d[:, 1])]
    # print(R2)
    # 结论等级参考值(由用户给定)
    consequent = [0, 0.5, 1]
    # 前提属性参考值集合
    ref = [R1, R2]
    # 样本维度既前提属性个数
    sum_m = len(ref)
    # 确定各前提属性的参考值个数
    sum_t = np.ones(sum_m)
    for j in range(0, sum_m):
        sum_t[j] = int(len(ref[j]))
    # 确定规则数目
    sum_l = 9
    sum_d = 3
    # 初始规则权重(由用户给定)
    theta = np.random.rand(sum_l)
    # 初始前提属性权重(由用户给定)
    delta = np.random.rand(sum_m)
    # 初始置信度,行数应与规则数相同，列数应与结论等级个数相同(由用户给定, 可由表格导入)
    belta = np.random.rand(sum_l, sum_d)
    for i in range(0, sum_l):   # 归一化
        belta[i] = belta[i] / sum(belta[i])
    belta_list = belta.flatten()
    P = list(theta) + list(delta) + list(belta_list)
    # print(theta, delta, belta.shape)

    # 提取数据
    input1 = np.zeros((69, 2))  # 定义
    output1 = np.zeros(69)
    for i in range(0, 25):
        # 存放数据
        input1[i] = low_d[5 * i]
        output1[i] = output[5 * i]  # 前125个数据的20%
    for i in range(0, 11):
        input1[25 + 4 * i: 25 + 4 * i + 4] = low_d[125 + 5 * i + 1: 125 + 5 * i + 5]  # 后58个数据的80%
        output1[25 + 4 * i: 25 + 4 * i + 4] = output[125 + 5 * i + 1: 125 + 5 * i + 5]  # 后58个数据的80%

    # 调用BRB函数，输入为数据和参数，输出belief为结论置信度分布，可作为上一层BRB的前提属性输入
    start = time.perf_counter()  # 开始时间
    data = [low_d, output, consequent]
    para = cmaes(data, ref, P)  # 训练集优化参数

    # 测试集
    data = [input1, output1, consequent]
    err = brb(data, ref, para)
    end = time.perf_counter()
    print('运行时间：%s' % (end - start))

    # 输出
    # output_excel = pd.DataFrame(test_out)
    # writer = pd.ExcelWriter('brnn2.xls')
    # output_excel.to_excel(writer, sheet_name='BPNN2', startcol=0, index=False)
    writer.save()
    # 画图
    plt.plot(output1)
    plt.show()