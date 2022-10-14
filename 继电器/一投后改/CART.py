"""
CART
"""
import numpy as np
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
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
    lamda = range(0, int(len(output) * 0.8), 1)
    dataset = input[lamda]  # 训练集
    labelset = output[lamda]  # 训练集标签
    test_dataset = np.delete(input, lamda, 0)    # 测试集
    test_labelset = np.delete(output, lamda, 0)  # 测试集标签

    # 标准化
    scaler = StandardScaler()
    scaler.fit(test_dataset)
    x_test_Standard = scaler.transform(test_dataset)
    scaler.fit(dataset)
    x_train_Standard = scaler.transform(dataset)
    scaler.fit(input)
    input_Standard = scaler.transform(input)
    # 运行
    # knn = KNeighborsClassifier(n_neighbors=6)
    CART = DecisionTreeRegressor()
    # 训练集
    CART.fit(x_train_Standard, labelset)

    # 测试集
    test_out = CART.predict(x_test_Standard)
    # test_out = CART.predict(input_Standard)
    # 训练集
    train_out = CART.predict(x_train_Standard)
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

    # print('测试误差为：', (sum((test_out - output) ** 2) / len(output)) ** 0.5)
    print('测试误差为：', (sum((test_out - test_labelset) ** 2) / len(test_labelset)) ** 0.5)
    print('拟合误差为：', (sum((train_out - labelset) ** 2) / len(train_out)) ** 0.5)
    # 画图
    plt.figure(1)
    plt.plot(output, '-v')
    plt.plot(list(train_out) + list(test_out), 'r')
    plt.title('test results')
    plt.figure(2)
    plt.plot(train_out, '-v')
    plt.plot(labelset, 'r')
    plt.title('train results')
    plt.show()

    # 存储
    output_excel = pd.DataFrame(list(train_out) + list(test_out))
    writer = pd.ExcelWriter('CART故障缺失.xlsx')
    output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    writer.save()