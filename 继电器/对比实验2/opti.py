from brb import BRB
import math
import numpy as np


def cmaes(data, ref, x):
    # ==================初始化==========================#
    sum_l = 9                  # 规则库数目
    sum_d = 4                  # 结论等级个数
    sum_t = 2                  # 属性个数
# -----------------------------cmaes---------------------------#
    FES = 0
    N = len(x)             # 维数
    xmean = x              # 初始值
    sigma = 0.5            # 步长(标准差)
    stopeval = 5          # 最大迭代次数   # stopfitness = 1e-10    # 截止误差
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
                arx[kk][k] = xmean[kk] + vert[kk][0]   # 均值+步长*正态分布C
            # print(arx[:, k])
    # % % % % % % % % % % % % % % % % % % 投影算子加约束 % % % % % % % % % % % % % %
            for i in range(0, sum_l):  # theta相对权重
                arx[i][k] = arx[i][k] / max(arx[: sum_l, k])
            for i in range(sum_l, sum_l + sum_t):  # delta相对权重
                if arx[i][k] > 1:
                    arx[i][k] = 1
                # arx[i][k] = arx[i][k] / max(arx[sum_l: (sum_l + sum_t), k])
            for i in range(0, sum_l):  # belta相对权重
                arx[((sum_l + sum_t) + i * sum_d): ((sum_l + sum_t) + i * sum_d) + sum_d, k] \
                    = arx[((sum_l + sum_t) + i * sum_d): ((sum_l + sum_t) + i * sum_d) + sum_d, k] / \
                      sum(arx[((sum_l + sum_t) + i * sum_d): ((sum_l + sum_t) + i * sum_d) + sum_d, k])
                # print(arx[(11 + i * sum_d): (11 + i * sum_d) + sum_d, k])
            # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            blank, arfitness[k] = BRB(data, ref, arx[:, k])  # 调用目标函数
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