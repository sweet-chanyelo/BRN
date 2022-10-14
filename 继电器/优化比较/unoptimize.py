""""""
from sklearn.model_selection import KFold
from scipy.optimize import leastsq
from ERR import error
import pandas as pd
import numpy as np
import math


def unoptimize(t, H):
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
    P = []
    err = []
    a = []
    input1 = np.zeros((84, 4))
    input2 = np.zeros((84, 4))
    output1 = np.zeros(84)
    for i in range(0, 20):
        # 存放数据
        input1[4 * i] = H[0][5 * i + 1]  # 前100个数据的80%
        input1[4 * i + 1] = H[0][5 * i + 2]
        input1[4 * i + 2] = H[0][5 * i + 3]
        input1[4 * i + 3] = H[0][5 * i + 4]
        input2[4 * i] = H[1][5 * i + 1]  # 前100个数据的80%
        input2[4 * i + 1] = H[1][5 * i + 2]
        input2[4 * i + 2] = H[1][5 * i + 3]
        input2[4 * i + 3] = H[1][5 * i + 4]
        output1[4 * i] = output[5 * i + 1]  # 前100个数据的80%
        output1[4 * i + 1] = output[5 * i + 2]
        output1[4 * i + 2] = output[5 * i + 3]
        output1[4 * i + 3] = output[5 * i + 4]
    for i in range(0, 16):
        input1[68 + i] = H[0][100 + 5 * i]  # 后80个数据的20%
        input2[68 + i] = H[1][100 + 5 * i]  # 后80个数据的20%
        output1[68 + i] = output[100 + 5 * i]  # 后80个数据的20%

    # 产生规则单元权重
    omega = np.random.randn(2)
    for i in range(0, len(omega)):
        if omega[i] > 1:
            omega[i] = 1
        if omega[i] <= 0:
            omega[i] = 0.00001
    P.append(omega)
    # 测试集
    for i in range(0, len(input1)):
        q, blank = error(omega, input1[i], input2[i], output1[i])  # 调用目标函数
        a.append(q)
    err.append(sum(a) / len(a))
    # print('第{}个规则单元测试误差为：{}统计参数值为：{}'.format(t, err[point], omega))
    point += 1
    # 交叉验证后的统计误差
    err_index = np.argsort(err)
    sta_err = sum(err) / len(err)
    # print('第{}个规则单元交叉验证后的统计误差{},最小误差为：{}，结构参数为：{}'.format(t, sta_err, err[err_index[0]], P[err_index[0]]))
    Para = P[err_index[0]]  # 最优结构参数
    belief = np.zeros((Data, 4))
    for i in range(0, Data):
        blank, belief[i] = error(Para, H[0][i], H[1][i], output[i])
    # print("最优结构参数下所有规则单元的输出置信度为：{}".format(belief))
    a = [[sta_err, sta_err, sta_err, sta_err]]
    res2 = np.r_[belief, a]  # 压缩
    return res2