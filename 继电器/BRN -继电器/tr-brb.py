"""
作者——张春潮
时间——2020/6/24
主题——航天继电器
        单层置信规则库
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import math
import time


start = time.perf_counter()  # 开始时间


def brb(p):
    # 给定置信规则库的初始值,确定各参数
    theta = p[0:729]  # 规则权重
    delta = p[729:735]  # 属性权重
    belta = np.array(p[735:]).reshape((729, 4))  # 置信度
    # 检验
    # print(belta)
    # 定义估计值
    output_ture = np.zeros(len(output))
    # 计算输入匹配度
    # alpha中保存每个前提属性相对于各自的参考值的匹配度，
    for l in range(0, len(input)):
        print(len(input))
        alpha = []
        for i in range(0, sum_m):
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
            alpha.append(alpha_i)
        # 检验
        # print(alpha)
        # 按照排列组合建立规则，确定最终的规则匹配度矩阵（L乘M）
        rule = list(itertools.product(alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))
        # 检验
        # 由输入匹配度确立规则激活权重
        alpha_k = np.ones(sum_l)
        omega = np.zeros(sum_l)
        # 生成激活权重
        for k in range(sum_l):
            for i in range(sum_m):
                alpha_k[k] = alpha_k[k] * (rule[k][i] ** float(delta[i]))
        for i in range(sum_l):
            omega[i] = (theta[i] * alpha_k[i] / sum(theta * alpha_k))
        # 规则的融合
        # 输出信度belta的正则化
        belief = np.zeros((len(input), sum_d))  # 定义输出置信度
        # 定义辅助变量
        a, b = np.ones(sum_d), 1
        for i in range(sum_d):
            for j in range(sum_l):
                a[i] = a[i] * (omega[j] * belta[j][i] + 1 - omega[j] * sum(belta[j]))
                b = np.prod(1 - omega * sum(belta[j]))
        mu = 1 / (sum(a) - (sum_d - 1) * b)
        for i in range(sum_d):
            belief[l][i] = mu * (a[i] - b) / (1 - mu * np.prod(1 - omega))
        output_ture[l] = np.mat(belief[l]) * np.mat(consequence).T  # 输出效用
    err = np.sum((output_ture - output) ** 2) / len(input)
    # print(err)
    return err, output_ture


def cmaes(x):
    # -----------------------------cmaes---------------------------#
    FES = 0
    N = len(x)             # 维数
    xmean = x              # 初始值
    sigma = 0.5            # 步长(标准差)
    stopeval = 5          # 最大迭代次数   # stopfitness = 1e-10    # 截止误差
    # 选择策略参数设置
    lamda = 20 + math.floor(3 * np.log(N))        # 后代的数量,种群大小
    mu = lamda / 2                                # 父代的数量
    weights = np.log(mu + 1 / 2) - np.log(range(1, int(mu)))  # 用于重组的父代矩阵?
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
    bestx = []    # 定义全局最优解
    while counteval < stopeval:  # 生成并评估所有后代, 第一代
        for k in range(0, lamda):  # 第一个后代
            vert = sigma * np.dot(B, np.multiply(D, np.random.rand(N, 1)))
            for kk in range(0, N):  # 设置上下界，参数约束
                arx[kk][k] = xmean[kk] + vert[kk][0]   # 均值+步长*正态分布C
            # print(arx[:, k])
    # % % % % % % % % % % % % % % % % % % 投影算子加约束 % % % % % % % % % % % % % %
            for i in range(0, sum_l):               # theta相对权重
                arx[i][k] = arx[i][k] / max(arx[: sum_l, k])
            for i in range(sum_l, sum_l + sum_m):   # delta相对权重
                if arx[i][k] > 1:
                    arx[i][k] = 1
                # arx[i][k] = arx[i][k] / max(arx[sum_l: (sum_l + sum_t), k])
            for i in range(0, sum_l):               # belta相对权重
                arx[((sum_l + sum_m) + i * sum_d): ((sum_l + sum_m) + i * sum_d) + sum_d, k] \
                    = arx[((sum_l + sum_m) + i * sum_d): ((sum_l + sum_m) + i * sum_d) + sum_d, k] / \
                      sum(arx[((sum_l + sum_m) + i * sum_d): ((sum_l + sum_m) + i * sum_d) + sum_d, k])
                # print(arx[(11 + i * sum_d): (11 + i * sum_d) + sum_d, k])
            # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            arfitness[k], blank = brb(list(arx[:, k]))  # 调用目标函数
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
        print(bestf)
    return bestx


def main():
    global ref, sum_m, sum_t, sum_l, sum_d, alpha, input, output, consequence
    # ============================由excel导入样本数据=======================#
    df = pd.read_excel('data3.xls', sheet_name='input-183', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    # 特征两两不重复组合
    ip = list(itertools.combinations(range(dim), 2))
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df.iat[i, j]
    # 导入标签数据
    df2 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    # print(df1, df2)
    # 第一个前提属性参考值(由用户给定)
    feature_1 = [2040.619, 2040.629, 2046.638]
    # 第二个前提属性参考值(由用户给定)
    feature_2 = [13.933, 14.270, 14.573]
    # 第三个前提属性参考值(由用户给定)
    feature_3 = [50.493, 50.85, 51.215]
    # 第四个前提属性参考值(由用户给定)
    feature_4 = [42.975, 43.55, 44.073]
    # 第五个前提属性参考值(由用户给定)
    feature_5 = [1.409, 1.416, 1.422]
    # 第六个前提属性参考值(由用户给定)
    feature_6 = [1.619, 1.628, 1.632]
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    sum_d = len(consequence)  # 结论等级个数
    # 前提属性参考值集合
    ref = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
    # 样本维度既前提属性个数
    sum_m = len(ref)
    # 确定各前提属性的参考值个数
    sum_t = np.ones(sum_m)
    for j in range(0, sum_m):
        sum_t[j] = int(len(ref[j]))
    # 规则库数目
    sum_l = int(np.prod(sum_t))
    # 初始规则权重(随机产生)
    theta = np.random.rand(sum_l)
    # 初始前提属性权重(随机产生)
    delta = np.random.rand(sum_m)
    # 初始置信度,行数应与规则数相同，列数应与结论等级个数相同(由用户给定, 可由表格导入)
    belta = np.random.rand(sum_l, sum_d)
    for i in range(0, sum_l):
        for j in range(0, sum_d):
            belta[i][j] = belta[i][j] / sum(belta[i])
    # 检验
    # print(belta.shape)
    # P为元胞阵，内含初始BRB参数
    belta_list = belta.flatten()
    P = list(theta) + list(delta) + list(belta_list)
    # 调用BRB函数，输入为数据和参数，输出belief为结论置信度分布
    p = cmaes(P)  # 存储历次结构参数
    err, output_ture = brb(p)
    # 检验
    end = time.perf_counter()
    print('运行时间：%s' % (end - start))
    output_excel = pd.DataFrame(output_ture)
    writer = pd.ExcelWriter('trbrb.xls')
    output_excel.to_excel(writer, sheet_name='trbrb', startcol=0, index=False)
    plt.plot(output_ture, 'g', lw=1)
    writer.save()
    plt.show()


if __name__ == '__main__':
    main()