"""
SVM
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    start = time.perf_counter()  # 开始时间
    out = []
    # 导入数据
    df = pd.read_excel('input2.xlsx', sheet_name='data', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        input[i - 1] = df.iloc[i]

    # 导入标签数据
    df2 = pd.read_excel('input2.xlsx', sheet_name='output', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iloc[i]
    # print(output)
    # 数据划分
    # lamda = range(0, len(output), 5)
    # dataset = np.delete(input, lamda, 0)  # 训练集
    # labelset = np.delete(output, lamda, 0)  # 训练集标签
    # test_dataset = input[lamda]  # 测试集
    # test_labelset = output[lamda]  # 测试集标签
    # print("*********", np.mat(dataset))

    lamda = [8, 12, 25, 82, 103, 105, 108, 119, 129, 131, 172, 173, 180, 191, 212]
    dataset = np.delete(input, lamda, 0)  # 训练集
    labelset = np.delete(output, lamda, 0)  # 训练集标签
    test_dataset = input[lamda]  # 测试集
    test_labelset = output[lamda]  # 测试集标签
    # 标准化
    scaler = StandardScaler()
    scaler.fit(test_dataset)
    x_test_Standard = scaler.transform(test_dataset)
    scaler.fit(dataset)
    x_train_Standard = scaler.transform(dataset)

    # 训练集
    clf = SVR(kernel='rbf', C=800)  # 核函数选择 rbf 径向基核函数
    # clf = svm.SVC(kernel='linear', C=800)  # 核函数选择 线性核函数
    # clf = svm.SVC(kernel='poly', C=800)  # 核函数选择 多项式核函数
    clf.fit(x_train_Standard, labelset)

    # 测试集
    test_out = clf.predict(x_test_Standard)
    # print('测试集输出为：', test_out)
    train_out = clf.predict(x_train_Standard)
    # print('训练集输出为：', train_out)
    # 输出
    end = time.perf_counter()  # 停止时间
    print('运行时间：%s' % (end - start))

    # temp = 0
    # for i in range(len(test_out)):
    #     if abs(test_out[i] - test_labelset[i]) < 0.1:
    #         temp += 1
    # print('测试误差为：', temp / len(test_out))
    # temp1 = 0
    # for i in range(len(train_out)):
    #     if abs(train_out[i] - labelset[i]) < 0.1:
    #         temp1 += 1
    # print('拟合误差为：', temp1 / len(train_out))

    print('测试误差为：', (sum((test_out - test_labelset) ** 2) / len(test_out)) ** 0.5)
    print('拟合误差为：', (sum((train_out - labelset) ** 2) / len(train_out)) ** 0.5)
    # 画图
    plt.figure(1)
    plt.plot(test_out, '-v')
    plt.plot(test_labelset, 'r')
    plt.title('test results')
    plt.figure(2)
    plt.plot(train_out, '-v')
    plt.plot(labelset, 'r')
    plt.title('train results')
    plt.show()
    # 存储
    # output_excel = pd.DataFrame(test_out)
    # writer = pd.ExcelWriter('RIMER15组输出.xlsx')
    # output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    # writer.save()