"""
历史最好误差， 0.009513476438695108
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as newPool
import itertools
from Belief_unit import Integration

if __name__ == '__main__':
    # ***********************************导入数据***************************************#
    Z = 21  # 定义子规则库个数
    df1 = pd.read_excel('BRB1.xlsx', "data", header=None)
    df2 = pd.read_excel('BRB2.xlsx', "data", header=None)
    df3 = pd.read_excel('BRB3.xlsx', "data", header=None)
    df4 = pd.read_excel('BRB4.xlsx', "data", header=None)
    df5 = pd.read_excel('BRB5.xlsx', "data", header=None)
    df6 = pd.read_excel('BRB6.xlsx', "data", header=None)
    df7 = pd.read_excel('BRB7.xlsx', "data", header=None)
    df8 = pd.read_excel('BRB8.xlsx', "data", header=None)
    df9 = pd.read_excel('BRB9.xlsx', "data", header=None)
    df10 = pd.read_excel('BRB10.xlsx', "data", header=None)
    df11 = pd.read_excel('BRB11.xlsx', "data", header=None)
    df12 = pd.read_excel('BRB12.xlsx', "data", header=None)
    df13 = pd.read_excel('BRB13.xlsx', "data", header=None)
    df14 = pd.read_excel('BRB14.xlsx', "data", header=None)
    df15 = pd.read_excel('BRB15.xlsx', "data", header=None)
    df16 = pd.read_excel('BRB16.xlsx', "data", header=None)
    df17 = pd.read_excel('BRB17.xlsx', "data", header=None)
    df18 = pd.read_excel('BRB18.xlsx', "data", header=None)
    df19 = pd.read_excel('BRB19.xlsx', "data", header=None)
    df20 = pd.read_excel('BRB20.xlsx', "data", header=None)
    df21 = pd.read_excel('BRB21.xlsx', "data", header=None)
    row, col = df1.shape
    brb1 = np.zeros((row - 1, col))
    brb2 = np.zeros((row - 1, col))
    brb3 = np.zeros((row - 1, col))
    brb4 = np.zeros((row - 1, col))
    brb5 = np.zeros((row - 1, col))
    brb6 = np.zeros((row - 1, col))
    brb7 = np.zeros((row - 1, col))
    brb8 = np.zeros((row - 1, col))
    brb9 = np.zeros((row - 1, col))
    brb10 = np.zeros((row - 1, col))
    brb11 = np.zeros((row - 1, col))
    brb12 = np.zeros((row - 1, col))
    brb13 = np.zeros((row - 1, col))
    brb14 = np.zeros((row - 1, col))
    brb15 = np.zeros((row - 1, col))
    brb16 = np.zeros((row - 1, col))
    brb17 = np.zeros((row - 1, col))
    brb18 = np.zeros((row - 1, col))
    brb19 = np.zeros((row - 1, col))
    brb20 = np.zeros((row - 1, col))
    brb21 = np.zeros((row - 1, col))
    for i in range(1, row):
        brb1[i - 1] = df1.iloc[i]
        brb2[i - 1] = df2.iloc[i]
        brb3[i - 1] = df3.iloc[i]
        brb4[i - 1] = df4.iloc[i]
        brb5[i - 1] = df5.iloc[i]
        brb6[i - 1] = df6.iloc[i]
        brb7[i - 1] = df7.iloc[i]
        brb8[i - 1] = df8.iloc[i]
        brb9[i - 1] = df9.iloc[i]
        brb10[i - 1] = df10.iloc[i]
        brb11[i - 1] = df11.iloc[i]
        brb12[i - 1] = df12.iloc[i]
        brb13[i - 1] = df13.iloc[i]
        brb14[i - 1] = df14.iloc[i]
        brb15[i - 1] = df15.iloc[i]
        brb16[i - 1] = df16.iloc[i]
        brb17[i - 1] = df17.iloc[i]
        brb18[i - 1] = df18.iloc[i]
        brb19[i - 1] = df19.iloc[i]
        brb20[i - 1] = df20.iloc[i]
        brb21[i - 1] = df21.iloc[i]
    # print(brb1, brb1.shape)

    # 输出
    df = pd.read_excel('input2.xlsx', "output", header=None)
    row, col = df.shape
    output = np.zeros(row - 1)
    for i in range(1, row):
        output[i - 1] = df.iloc[i]
    # print(output)
    lamda = len(output)
    # 结论等级参考值(由用户给定)
    consequence = [0, 0.25, 0.5, 1]
    output_ture = np.zeros(lamda)
    error = []
    errr = []
    err_index = []
    error_var = []

    # BRB = [brb1, brb2, brb3, brb4, brb5, brb6, brb7, brb8, brb9, brb10, brb11,
    #        brb12, brb13, brb14, brb15, brb16, brb17, brb18, brb19, brb20, brb21]
    BRB = [brb1, brb2, brb3, brb4, brb5, brb6, brb7, brb8, brb9]
    # 各个子规则库的输出精度
    for J in BRB:
        for i in range(lamda):
            output_ture[i] = np.dot(J[i], consequence)
        errr.append((np.sum((output_ture - output) ** 2) / lamda))
    err_index = np.argsort(errr)
    error.append(errr[err_index[0]])
    # error_var.append(np.var(errr))
    print('子置信规则库的测试误差为：', errr)
    # —————————————————并行处理—————————————————————#
    # belief = [brb1, brb2, brb3, brb4, brb5, brb6, brb7, brb8, brb9, brb10, brb11,
    #           brb12, brb13, brb14, brb15, brb16, brb17, brb18, brb19, brb20, brb21]
    belief = [brb1, brb2, brb3, brb4, brb5, brb6, brb7, brb8, brb9]
    ip2 = list(itertools.combinations(range(len(belief)), 2))
    num_j = 9
    layer = 1
    best_err = 1
    best_belief = np.zeros((lamda, col))
    while layer < 10:
        print('.............第{}层.................'.format(layer))
        err = np.zeros(len(ip2))
        Sample = []
        for i in range(len(ip2)):
            Sample.append([belief[ip2[i][0]], belief[ip2[i][1]]])
        p2 = newPool(2)
        res2 = p2.map(Integration, range(len(ip2)), Sample)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@数据更新@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
        for i in range(len(res2)):
            err[i] = res2[i][1]
        err_index = np.argsort(err)  # 误差排序
        error.append(err[err_index[0]])
        print('第{}层最小的误差为{}'.format(layer, err[err_index[0]]))
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@筛选@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
        belief = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        err_var = np.zeros(num_j)
        for i in range(num_j):
            belief[i] = res2[err_index[i]][0]
            err_var[i] = err[err_index[i]]
        error_var.append(np.var(err_var))
        #存储
        if layer == 1:  # 初始化最优解
            best_belief = belief[0]
            best_err = err[err_index[0]]
        elif err_var[0] < best_err:  # 更新最优解
            best_belief = belief[0]
            best_err = err_var[0]
        # print('误差索引为', err_index, '误差方差为', np.var(err_var))
        # 并行计算
        ip2 = list(itertools.combinations(range(num_j), 2))
        num_j = 21  # 更新下一层中置信单元的数量
        layer += 1
    error_index = np.argsort(error)
    print('全局最小误差为：', error[error_index[0]])
    # 画图
    x = np.arange(216)
    plt.figure(1)  # 误差变化图
    plt.plot(error, '-.*')
    plt.title('Error')

    plt.figure(2)  # 仿真输出
    for i in range(0, lamda):  # 输出效用
        output_ture[i] = np.dot(belief[0][i], consequence)

    index = range(0, int(len(output) * 0.8), 1)
    train_y = output_ture[index]
    test_y = np.delete(output, index, 0)
    test_y_wave = np.delete(output_ture, index, 0)  # 测试集
    print((np.sum((test_y_wave - test_y) ** 2) / len(test_y)) ** 0.5)

    # index = range(0, lamda, 5)
    # train_y = np.delete(output, index, 0)
    # test_y_wave = output_ture[index]  # 测试集
    # test_y = output[index]

    # index = [8, 12, 25, 82, 103, 105, 108, 119, 129, 131, 172, 173, 180, 191, 212]
    # train_y = np.delete(output, index, 0)
    # test_y_wave = output_ture[index]  # 测试集
    # test_y = output[index]
    # print((np.sum((test_y_wave - test_y) ** 2) / 20) ** 0.5)

    plt.plot(output_ture, color='g')
    #     # print(output_ture)
    plt.plot(output, color='r')
    plt.title('Simulated results')

    fig3 = plt.figure(3)  # 误差方差变化图
    plt.plot(error_var, '-.*')
    plt.yscale('log')  # 科学计数法
    plt.title('The varance of error')

    fig4 = plt.figure(4)  # 输出置信分布
    # print(best_belief[:, 0])
    ax1 = plt.subplot(311)
    plt.bar(x, best_belief[:, 0])
    ax2 = plt.subplot(312)
    plt.bar(x, best_belief[:, 1])
    ax3 = plt.subplot(313)
    plt.bar(x, best_belief[:, 2])

    fig4.tight_layout()
    plt.show()
    # 存储
    output_excel = pd.DataFrame(output_ture)
    writer = pd.ExcelWriter('RIMER-PCA故障缺失.xlsx')
    output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    writer.save()
