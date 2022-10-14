
import numpy as np


def error(omega, x1, x2, y):
    # print('***', x1, x2, y)
    sum_d = len(x1)      # 结论等级个数 =4
    sum_l = 2            # 规则数目=2
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    belta = [x1, x2]  # 定义置信分布
    m = np.ones((sum_l, sum_d))  # 定义基本概率质量(2,4)
    for k in range(sum_l):
        for l in range(sum_d):
            m[k][l] = omega[k] * belta[k][l]
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
    m_i = np.zeros(sum_d)
    # 定义输出置信度
    belief = np.zeros(sum_d)
    # 规则融合
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
        m_i[i] = (m[0][i] * m[1][i] + m[0][i] * m_d[1] + m_d[0] * m[1][i]) / coeff
    m_wave_i = (m_wave[0] * m_wave[1] + m_wave[0] * m_hat[1] + m_hat[0] * m_wave[1]) / coeff
    m_hat_i = (m_hat[0] * m_hat[1]) / coeff
    m_d_i = m_hat_i + m_wave_i
    # 输出最终基本概率质量
    # print(m_i)
    # 基本概率质量转化为置信度
    for d in range(0, sum_d):
        belief[d] = m_i[d] / (1 - m_hat_i)
    # 计算效用与误差
    # eval_y[m] = (np.mat(belief) * np.mat(consequence).T).tolist()  # 输出效用
    eval_y = (np.mat(belief) * np.mat(consequence).T).tolist()  # 输出效用
    err = (eval_y - y) ** 2
    return err[0][0], belief