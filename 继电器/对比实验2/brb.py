"""
BRB设计
"""
from ER import agg
import numpy as np
import itertools


def BRB(data, ref, p):
    """
    data为1乘M矩阵的某一个输入样本数据，具有M个前提属性
    """
# **************************初始信息********************************#
    train_u = data[0]        # 训练集
    train_z = data[1]        # 训练集标签
    eval_z = np.zeros(len(train_z))      # 定义模型估计值
    consequence = data[2]    # 评估等级
    # 给定置信规则库的初始值,确定各参数
    theta = p[0:9]            # 规则权重
    delta = p[9:11]            # 属性权重
    belta = np.array(p[11:]).reshape((9, 4))    # 置信度
    # print(len(p), theta, delta, belta)
    sum_t = len(train_u[0])        # 前提属性个数
    sum_l, sum_d = belta.shape     # 规则个数与评估等级个数
    # 确定各前提属性的参考值个数
    sum_m = np.ones(sum_t)
    for j in range(0, sum_t):
        sum_m[j] = int(len(ref[j]))
    # 前提属性的归一化
    # delta_hat = np.zeros(sum_t)
    # for i in range(sum_t):
    #     delta_hat[i] = delta[i] / max(delta)
    err = 0
    belief = np.zeros((len(train_u), sum_d))
# ————————————————输入匹配度—————————-—————#
    for l in range(0, len(train_u)):
        # alpha中保存每个前提属性相对于各自的参考值的匹配度，
        alpha = [[], []]
        for i in range(0, sum_t):
            # 前提属性的参考值个数
            r = int(sum_m[i])
            alpha_i = np.zeros(r)
            for j in range(0, r):
                if j + 1 != r and ref[i][j] <= train_u[l][i] < ref[i][j + 1]:
                    alpha_i[j] = (ref[i][j + 1] - train_u[l][i]) / (ref[i][j + 1] - ref[i][j])
                    alpha_i[j + 1] = (train_u[l][i] - ref[i][j]) / (ref[i][j + 1] - ref[i][j])
                elif j + 1 == r and ref[i][j] <= train_u[l][i]:
                    alpha_i[j] = 1
                elif j + 1 == r and ref[i][0] >= train_u[l][i]:
                    alpha_i[0] = 1
            alpha[i] = alpha_i
        # 按照排列组合建立规则，确定最终的规则匹配度矩阵（L乘M）
        rule = list(itertools.product(alpha[0], alpha[1]))
        # 检验
        # print(rule)
# ==========================规则激活权重======================#
        alpha_k = np.ones(sum_l)
        omega = np.zeros(sum_l)
        pointer = []
        # 生成激活权重
        for k in range(sum_l):
            for i in range(sum_t):
                alpha_k[k] = alpha_k[k] * (rule[k][i] ** float(delta[i]))
            if alpha_k[k] > 0:
                pointer.append(k)
        for i in range(len(pointer)):
            if sum(theta * alpha_k) == 0:
                omega[pointer[i]] = 0
                print(theta)
            else:
                omega[pointer[i]] = (theta[pointer[i]] * alpha_k[pointer[i]] / sum(theta * alpha_k))
        # print(pointer)
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&规则的融合&&&&&&&&&&&&&&&&&&&&&&&#
        m = np.ones((sum_l, sum_d))
        for k in range(sum_l):
            m[k] = omega[k] * belta[k]
        # print(m)
        belief[l] = agg(m, omega, belta, pointer)
        # print(belief)                           # 检验置信分布
# =================================外准则评估===========================#
        eval_z[l] = np.mat(belief[l]) * np.mat(consequence).T  # 输出效用
        err = np.sum((eval_z - train_z) ** 2) / len(train_z)
    # print(data[0].shape)
    # if len(data[0]) < 50:  # 随数据量而变
    #     return err
    # elif len(data[0]) > 60:
    return belief, err