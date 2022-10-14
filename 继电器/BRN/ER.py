"""

"""
import numpy as np


def agg(m, sum_l, sum_d, omega, belta, pointer):
    """
    :param m: 基本概率质量
    :param m_d:剩余基本概率质量
    :param m_hat:剩余激活权重，即不完整性
    :param m_wave:剩余输出信度，即无知性
    :return: belief是规则融合后的最终输出结果的置信度
    """
    # m = np.array(M)
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
