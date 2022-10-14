""""""
from sklearn.model_selection import KFold
from scipy.optimize import leastsq
from ERR import error
import pandas as pd
import numpy as np
import math


def integration(t, H):
    """
    """
    # 导入标签数据
    df2 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    # print(output)
    # 初始化
    Data, sum_d = H[0].shape   # 数据量，结论等级个数
    # print("规则单元{}的输入参数：{}, {}".format(t, H[0], H[1]))
    point = 0
    # 交叉验证
    KF = KFold(5)
    P = np.zeros((5, 2))
    err = []
    for train_index, test_index in KF.split(H[0], H[1], output):
        train_x1 = H[0][train_index]   # 训练集索引
        train_x2 = H[1][train_index]   # 训练集索引
        train_y = output[train_index]  # 训练集标签索引
        test_x1 = H[0][test_index]   # 测试集
        test_x2 = H[1][test_index]   # 测试集
        test_y = output[test_index]  # 测试集标签
        # 产生规则单元权重
        omega = np.random.rand(2)
        a = []
        # -----------------------------cmaes---------------------------#
        FES = 0
        N = len(omega)  # 维数
        xmean = omega   # 初始值
        sigma = 0.5     # 步长(标准差)
        stopeval = 10   # 最大迭代次数   # stopfitness = 1e-10    # 截止误差
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
        temp = np.zeros(len(train_x1))
        while counteval < stopeval:  # 生成并评估所有后代, 第一代
            for k in range(0, lamda):  # 第一个后代
                vert = sigma * np.dot(B, np.multiply(D, np.random.randn(N, 1)))
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
                for j in range(0, len(train_x1)):
                    temp[j], blank = error(arx[:, k], train_x1[j], train_x2[j], train_y[j])  # 调用目标函数
                arfitness[k] = np.sum(temp) / len(train_x1)
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
                FES = counteval
        # print('****', bestx, bestf)
    # 该训练集的最优参数取均值
        P[point] = bestx
        # 测试集
        for j in range(0, len(test_x1)):
            q, blank = error(P[point], test_x1[j], test_x2[j], test_y[j])  # 调用目标函数
            a.append(q)
        err.append(sum(a) / len(a))
        # print('第{}个规则单元测试误差为：{}统计参数值为：{}'.format(t, err[point], P[point]))
        point += 1
    # 交叉验证后的统计误差
    err_index = np.argsort(err)
    sta_err = sum(err) / len(err)
    print('第{}个规则单元交叉验证后的统计误差{},最小误差为：{}，结构参数为：{}'.format(t, sta_err, err[err_index[0]], P[err_index[0]]))
    Para = P[err_index[0]]  # 最优结构参数
    belief = np.zeros((Data, 4))
    for i in range(0, Data):
        blank, belief[i] = error(Para, H[0][i], H[1][i], output[i])
    # print("最优结构参数下所有规则单元的输出置信度为：{}".format(belief))
    a = [[sta_err, sta_err, sta_err, sta_err]]
    res2 = np.r_[belief, a]  # 压缩
    return res2