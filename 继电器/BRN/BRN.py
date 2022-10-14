"""
作者——张春潮
时间——2020/06/08
主题——置信规则网络BRN
    子BRB设计
    两输入
变化——
    1）采用pandas
    2）调整结构，分页结构
"""
from func import main
from rule_unit import integration
from pathos.multiprocessing import ProcessingPool as newPool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import time


if __name__ == '__main__':
    # ============================由excel导入样本数据=======================#
    df = pd.read_excel('data2.xlsx', sheet_name='input', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    # 特征两两不重复组合
    ip = list(itertools.combinations(range(dim), 2))
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df.iat[i, j]
    # print(input)
# ————————————————专家知识——————————---——————#
    # 初始规则权重(由用户给定)
    df = pd.read_excel('data2.xlsx', sheet_name='theta', header=None)
    height, width = df.shape
    theta = np.zeros(height - 1)
    for i in range(1, height):
        theta[i - 1] = df.iat[i, 0]
    # 初始前提属性权重(由用户给定)
    df = pd.read_excel('data2.xlsx', sheet_name='delta', header=None)
    height, width = df.shape
    delta = np.zeros(height - 1)
    for i in range(1, height):
        delta[i - 1] = df.iat[i, 0]
    # 初始置信度,行数应与规则数相同，列数应与结论等级个数相同(由用户给定, 可由表格导入)
    df = pd.read_excel('data2.xlsx', sheet_name='belta', header=None)
    height, width = df.shape
    belta = np.zeros((height - 1, width))
    for i in range(1, height):
        for j in range(0, width):
            belta[i - 1][j] = df.iat[i, j]
    # print(theta, delta, belta.shape)
    # P为元胞阵，内含初始BRB参数
    p = [theta, delta, belta]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%并行处理置信规则库%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    start = time.perf_counter()  # 开始时间
    # 多进程处理
    p1 = newPool(3)
    P = [p, p, p, p, p, p]
    res = p1.map(main, range(len(ip)), P)
    # p1.close()
    # p1.join()
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&外准则筛选&&&&&&&&&&&&&&&&&&&&&&&&#
    err = np.zeros(len(ip))
    belief = [[], [], [], [], [], []]
    for i in range(0, len(ip)):
        belief[i] = res[i][: N-1, :]
        err[i] = res[i][N-1, 0]
    # print(err, belief)
    err_index = np.argsort(err)  # 子置信规则库统计误差排序
    print(err_index)
    best_err = err[err_index[0]]  # 最优误差
    # ----------------------置信规则网络-------------------------------#
    lamda = 4   # 固定个数选拔
    while 1:
        h = [[], [], [], []]  # 定义规则单元
        #  选拔
        for i in range(0, lamda):   # 取前lamda个规则单元
            h[i] = belief[err_index[i]]
        ip2 = list(itertools.combinations(range(lamda), 2))  # 两两组合
        # print(ip2)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%并行处理置信规则网络%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
        # 多进程处理
        H = [[], [], [], [], [], []]
        for i in range(0, len(ip2)):
            H[i] = [h[ip[i][0]], h[ip[i][1]]]  # 不同单元的置信分布参数两两组合
        p2 = newPool(3)
        res2 = p2.map(integration, range(len(ip2)), H)
        # p1.close()
        # p1.join()
        # p2.close()
        # p2.join()
        #  返回误差及置信度
        # print(res2)
        err = np.zeros(len(ip2))
        belief = [[], [], [], [], [], []]
        for i in range(0, len(ip2)):
            belief[i] = res2[i][: N - 1, :]
            err[i] = res2[i][N - 1, 0]
        err_index = np.argsort(err)  # 子置信规则库统计误差排序
        print(err_index)
        # 截止条件
        if err[err_index[0]] < best_err:
            best_err = err[err_index[0]]
        elif err[err_index[0]] >= best_err:
            break
    end = time.perf_counter()   # 停止时间
    print('运行时间：%s' % (end - start))
    # 画图
    plt.show()






