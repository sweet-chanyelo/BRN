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
import time


start = time.perf_counter()  # 开始时间


def brb():
    """
    data为1乘M矩阵的某一个输入样本数据，具有M个前提属性
    n为含M元素的元胞阵，每个元素为1乘x矩阵，x为某个前提属性的参考值
    sum_m为前提属性个数
    sum_l为规则个数
    sum_d为结论等级个数
    p为含3个元素的元胞阵
      p1元素为1乘L的规则权重矩阵
      p2元素为1乘M的前提属性权重矩阵
      p3元素为L乘d的置信度矩阵
    """
    # 给定置信规则库的初始值,确定各参数
    theta = p[0][0]
    # 检验
    # print(theta)
    delta = p[1][0]
    # 检验
    # print(delta)
    belta = p[2]
    # 检验
    # print(belta)
    delta_hat = np.zeros(sum_m)
    # 前提属性的归一化
    for i in range(sum_m):
        delta_hat[i] = delta[i] / max(delta)
    # 按照排列组合建立规则，确定最终的规则匹配度矩阵（L乘M）
    rule = list(itertools.product(alpha[0], alpha[1]))
    # 检验
    # print(rule)
    # 由输入匹配度确立规则激活权重
    alpha_k = np.ones(sum_l)
    omega = np.zeros(sum_l)
    # 生成激活权重
    for k in range(sum_l):
        for i in range(sum_m):
            alpha_k[k] = alpha_k[k] * (rule[k][i] ** float(delta_hat[i]))
    for i in range(sum_l):
        omega[i] = (theta[i] * alpha_k[i] / sum(theta * alpha_k))
    # 规则的融合
    # 输出信度belta的正则化
    belief = np.zeros(sum_d)  # 定义输出置信度
    # 定义辅助变量
    a, b = np.ones(sum_d), 1
    for i in range(sum_d):
        for j in range(sum_l):
            a[i] = a[i] * (omega[j] * belta[j][i] + 1 - omega[j] * sum(belta[j]))
            b = np.prod(1 - omega * sum(belta[j]))
    mu = 1 / (sum(a) - (sum_d - 1) * b)
    for i in range(sum_d):
        belief[i] = mu * (a[i] - b) / (1 - mu * np.prod(1 - omega))
    return belief


def main():
    global ref, sum_m, sum_t, sum_l, sum_d, p, alpha
    # ============================由excel导入样本数据=======================#
    df = pd.read_excel('data3.xls', sheet_name='train-in', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    # 特征两两不重复组合
    ip = list(itertools.combinations(range(dim), 2))
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df.iat[i, j]
    # 导入标签数据
    df2 = pd.read_excel('data3.xls', sheet_name='train-out', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    # print(df1, df2)
    # 前提属性组成的一个样本(由用户给定，可由excel导入)
    # 定义估计值
    ouput_ture = np.zeros((len(output), 1))
    # x = np.linspace(7.00, 12.00, 2007)
    # 画观测数据的图
    plt.plot(output, 'r', lw=1)
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
    delta = np.random.rand(dim)
    # 初始置信度,行数应与规则数相同，列数应与结论等级个数相同(由用户给定, 可由表格导入)
    belta = np.random.rand((sum_l, sum_d))
    # 检验
    # print(tuple_p3)
    # P为元胞阵，内含初始BRB参数
    p = [theta, delta, belta]
    for l in range(0, len(input)):
        # 计算输入匹配度
        # alpha中保存每个前提属性相对于各自的参考值的匹配度，
        alpha = [[], []]
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
            alpha[i] = alpha_i
        # 检验
        # print(alpha)
        # 调用BRB函数，输入为数据和参数，输出belief为结论置信度分布，可作为上一层BRB的前提属性输入
        belief = brb()
        # 检验
        # print(belief)
        # 输出效用
        ouput_ture[l] = np.dot(belief, consequence)
        # 输出
        # print('石油泄漏程度：[无泄漏：{}]，[泄漏很小：{}]，[泄漏中等：{}],[泄漏较大：{}],[泄漏很大：{}], [效用:{}]'
        #       .format(belief[0], belief[1], belief[2], belief[3], belief[4], ouput))
        # print(ouput_ture)
        # 画图
    end = time.perf_counter()
    print('运行时间：%s' % (end - start))
    plt.plot(ouput_ture, 'g', lw=1)
    plt.show()


if __name__ == '__main__':
    main()