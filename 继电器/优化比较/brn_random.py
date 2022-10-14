"""
    规则参数随机给定，不进行优化
    BRN1
"""
from opti import cmaes
from brb import BRB
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


def main(t):
    # 由excel导入样本数据
    df1 = pd.read_excel('data3.xls', sheet_name='input-183', header=None)
    # 样本量与数据维度
    N, dim = df1.shape
    # 特征两两不重复组合
    ip = list(itertools.combinations(range(dim), 2))
    sd = df1.iloc[:, [ip[t][0], ip[t][1]]]
    height, width = sd.shape
    input = np.zeros((height - 1, width))
    for i in range(1, height):
        for j in range(0, width):
            input[i - 1][j] = sd.iat[i, j]
    # 导入标签数据
    df2 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    # print(df1, df2)
# ————————————————专家知识——————————---——————#
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
    sum_d = len(consequence)         # 结论等级个数
    # 前提属性参考值集合
    Ax = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
    # 属性两两不重复组合
    ref = [Ax[ip[t][0]], Ax[ip[t][1]]]
    # print(ref)
    # 子置信规则库前提属性个数(由用户给定)
    sum_t = 2
    sum_l = 9  # 规则库数目
    L = 135    # 规则总数，不是通用的，还要改
    # print(dim, L)
    # 初始规则权重(随机产生)
    theta = np.random.rand(sum_l)
    # 初始前提属性权重(随机产生)
    delta = np.random.rand(sum_t)
    # 初始置信度,行数应与规则数相同，列数应与结论等级个数相同(由用户给定, 可由表格导入)
    belta = np.random.rand(sum_l, sum_d)
    for i in range(0, sum_l):
        for j in range(0, sum_d):
            belta[i][j] = belta[i][j] / sum(belta[i])
    # print(theta, delta, belta.shape)
    # P为元胞阵，内含初始BRB参数
    p = [theta, delta, belta]
    # print(p)
    # 结构参数分组
    belta_list = belta.ravel(-1)
    P = list(theta) + list(delta) + list(belta_list)
# ************************************数据分组******************************#
    data = [input, output, consequence]
    belief, err = BRB(data, ref, P)
    # print("最优结构参数下所有数据的输出置信度为：%s" % (belief))
    a = [[err, err, err, err]]
    res = np.r_[belief, a]  # 压缩
    return res