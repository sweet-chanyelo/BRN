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
    df = pd.read_excel('data3.xls', sheet_name='input-183', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    # 特征两两不重复组合
    ip1 = list(itertools.combinations(range(dim), 2))
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df.iat[i, j]
    # print(input)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%并行处理置信规则库%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    start = time.perf_counter()  # 开始时间
    # 多进程处理
    p1 = newPool(3)
    res = p1.map(main, range(len(ip1)))
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&外准则筛选&&&&&&&&&&&&&&&&&&&&&&&&#
    err = np.zeros(len(ip1))
    belief = []
    best_belief = []
    for i in range(0, len(ip1)):
        belief.append(res[i][: N-1, :])
        err[i] = res[i][N-1, 0]
    # print('统计误差{}， 最优结构参数下所有数据的置信分布输出{}'.format(err, belief))
    err_index = np.argsort(err)  # 子置信规则库统计误差排序
    errr = []
    best_err = err[err_index[0]]  # 最优误差
    errr.append(err[err_index[0]])
    # ----------------------置信规则网络-------------------------------#
    lamda = 6   # 固定个数选拔
    layer = 0
    best_err_index = 0
    while layer < 20:
        print("第{}层".format(layer))
        h = []  # 定义规则单元
        #  选拔
        for i in range(0, lamda):   # 取前lamda个规则单元
            h.append(belief[err_index[i]])
        ip2 = list(itertools.combinations(range(lamda), 2))  # 两两组合
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%并行处理置信规则网络%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
        # 多进程处理
        H = []
        for i in range(0, len(ip2)):
            H.append([h[ip2[i][0]], h[ip2[i][1]]])  # 不同单元的置信分布参数两两组合
        p2 = newPool(3)
        res2 = p2.map(integration, range(len(ip2)), H)
        #  返回误差及置信度
        # print(res2)
        err = np.zeros(len(ip2))
        belief = []
        for i in range(0, len(ip2)):
            belief.append(res2[i][: N - 1, :])
            err[i] = res2[i][N - 1, 0]
        err_index = np.argsort(err)  # 子置信规则库统计误差排序
        # 截止条件
        if err[err_index[0]] < best_err:
            best_err = err[err_index[0]]
            best_err_index = err_index[0]
            best_belief = belief
        errr.append(err[err_index[0]])
        # elif err[err_index[0]] >= best_err:
        #     break
        layer += 1
    # p1.close()
    # p1.join()
    # p2.close()
    # p2.join()
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    output_ture = []
    for i in range(0, len(input)):
        result = np.mat(best_belief[best_err_index][i]) * np.mat(consequence).T
        output_ture.append(result[0, 0])   # 输出效用
    end = time.perf_counter()   # 停止时间
    # 输出
    print('运行时间：%s' % (end - start))
    print('统计泛化误差最小的规则单元', best_err)
    print(errr)
    # 画图
    output_excel = pd.DataFrame(output_ture)
    writer = pd.ExcelWriter('brn.xls')
    output_excel.to_excel(writer, sheet_name='brn', startcol=0, index=False)
    writer.save()

    plt.show()






