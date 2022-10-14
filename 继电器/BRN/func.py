"""

"""
from opti import cmaes
from brb import BRB
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


def main(t, p):
    # 由excel导入样本数据
    df1 = pd.read_excel('data2.xlsx', sheet_name='input', header=None)
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
    df2 = pd.read_excel('data2.xlsx', sheet_name='output', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    # print(df1, df2)
# ————————————————专家知识——————————---——————#
    # 第一个前提属性参考值(由用户给定)
    feature_1 = [0, 0.5, 1]
    # 第二个前提属性参考值(由用户给定)
    feature_2 = [0, 0.5, 1]
    # 第三个前提属性参考值(由用户给定)
    feature_3 = [0, 0.5, 1]
    # 第四个前提属性参考值(由用户给定)
    feature_4 = [0, 0.5, 1]
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.5, 1]
    # 前提属性参考值集合
    Ax = [feature_1, feature_2, feature_3, feature_4]
    # 子置信规则库前提属性个数(由用户给定)
    sum_t = 2
    # 属性两两不重复组合
    ref = [Ax[ip[t][0]], Ax[ip[t][1]]]
    # print(ref)
    # 确定各前提属性的参考值个数
    sum_m = np.ones(sum_t)
    for j in range(0, sum_t):
        sum_m[j] = int(len(ref[j]))
    sum_l = int(np.prod(sum_m))  # 规则库数目
    # 结构参数分组
    theta = p[0][sum_l*t:sum_l*(t+1)]      # 规则权重
    delta = p[1][sum_t*t:sum_t*(t+1)]      # 属性权重
    belta = p[2][sum_l*t:sum_l*(t+1), :]   # 置信度
    belta_list = belta.ravel(-1)
    # print(theta, delta, belta_list)
    P = list(theta) + list(delta) + list(belta_list)
# ************************************数据分组******************************#
    # data = pd.concat([df1, df2], axis=1)  # 输入输出数据合并
    err = np.zeros(height-1)                 # 定义交叉验证误差
    belief = np.zeros((height, 3))         # 定义交叉验证置信度
    p = np.zeros((height-1, len(P)))         # 定义交叉验证结构参数
    point = 0
    # 交叉验证
    LOOCV = LeaveOneOut()
    for train_index, test_index in LOOCV.split(input, output):
        train_x = input[train_index]     # 训练集
        train_y = output[train_index]    # 训练集标签
        test_x = input[test_index]       # 测试集
        test_y = output[test_index]      # 测试集标签
        data = [train_x, train_y, consequence]
        # 调用BRB函数，输入为数据和参数，输出belief为结论置信度分布
        p[point] = cmaes(data, ref, P)         # 存储历次结构参数
        data = [test_x, test_y, consequence]
        err[point] = BRB(data, ref, p[point])  # 存储历次测试输出误差与输出置信度
        point += 1
    err_index = np.argsort(err)    # 历次误差排序
    sta_err = sum(err) / len(err)  # 统计泛化误差（11次测试误差）
    print("子置信规则库{}的统计误差为：{},最小误差为：{}".format(t, sta_err, err[err_index[0]]))
    data = [input, output, consequence]
    belief = BRB(data, ref, p[err_index[0]])
    # print("最优结构参数下所有数据的输出置信度为：%s" % (belief))
    a = [[sta_err, sta_err, sta_err]]
    res = np.r_[belief, a]  # 压缩
    return res