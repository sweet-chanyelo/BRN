import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    start = time.perf_counter()  # 开始时间
    out = []
    # 导入数据
    df = pd.read_excel('data4.xls', sheet_name='input-183', header=None)
    # 样本量与数据维度
    N, dim = df.shape
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df.iat[i, j]
    dataset = []  # 训练集
    test_dataset = []  # 测试集
    for i in range(0, 25):
        test_dataset.append(input[5 * i])
        # 存放数据
        dataset.append(input[5 * i + 1])  # 前100个数据的80%
        dataset.append(input[5 * i + 2])
        dataset.append(input[5 * i + 3])
        dataset.append(input[5 * i + 4])
    for i in range(0, 11):
        dataset.append(input[125 + 5 * i])  # 后80个数据的20%
        # 测试数据
        test_dataset.append(input[125 + 5 * i + 1])
        test_dataset.append(input[125 + 5 * i + 2])
        test_dataset.append(input[125 + 5 * i + 3])
        test_dataset.append(input[125 + 5 * i + 4])
    # 导入标签数据
    df2 = pd.read_excel('data4.xls', sheet_name='output-183', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    labelset = []
    test_labelset = []
    for i in range(0, 25):
        test_labelset.append(output[5 * i])
        # 存放数据
        labelset.append(output[5 * i + 1])  # 前100个数据的80%
        labelset.append(output[5 * i + 2])
        labelset.append(output[5 * i + 3])
        labelset.append(output[5 * i + 4])
    for i in range(0, 11):
        labelset.append(output[125 + 5 * i])  # 后80个数据的20%
        # 测试数据
        test_labelset.append(output[125 + 5 * i + 1])
        test_labelset.append(output[125 + 5 * i + 2])
        test_labelset.append(output[125 + 5 * i + 3])
        test_labelset.append(output[125 + 5 * i + 4])
    # print("*********", np.mat(dataset))
    x = np.mat(dataset)
    y = 100 * np.mat(labelset)
    y = np.array(y).astype('int')
    y = y[0]
    # print("$$$$$$", y)
    clf = svm.SVC(kernel='rbf', C=800)  # 核函数选择rbf
    clf.fit(x, y)

    z = np.mat(test_dataset)  # 测试集
    # print("%%%%%", np.mat(test_dataset).shape)
    out = clf.predict(z) * 0.01

    end = time.perf_counter()  # 停止时间
    # 输出到表格
    print('运行时间：%s' % (end - start))
    output_excel = pd.DataFrame(out)
    writer = pd.ExcelWriter('svm.xls')
    output_excel.to_excel(writer, sheet_name='svm', startcol=0, index=False)
    writer.save()
    # 画图
    plt.figure()
    plt.plot(out)
    print(sum((out - test_labelset) ** 2) / len(out))
    plt.ylim(0, 1)  # y幅度
    plt.show()
