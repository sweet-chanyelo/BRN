import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
def parameter_initialization(x, y, z):
    # 隐层阈值
    value1 = np.random.rand(1, y).astype(np.float64)  # (0,1)之间

    # 输出层阈值
    value2 = np.random.rand(1, z).astype(np.float64)  # (0,1)之间
    # print(value1, value2)
    # 输入层与隐层的连接权重
    weight1 = np.random.rand(x, y).astype(np.float64)

    # 隐层与输出层的连接权重
    weight2 = np.random.rand(y, z).astype(np.float64)
    # print(weight1, weight2)
    return weight1, weight2, value1, value2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def trainning(dataset, labelset, weight1, weight2, value1, value2):
    # x为步长
    x = 0.01
    for i in range(len(dataset)):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.float64)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)
        # 隐层输入
        input1 = np.dot(inputset, weight1).astype(np.float64)
        # 隐层输出
        output2 = sigmoid(input1 - value1).astype(np.float64)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)
        # 输出层输出
        output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵运算表示
        a = np.multiply(output3, 1 - output3)
        g = np.multiply(a, outputset - output3)
        b = np.dot(g, np.transpose(weight2))
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)

        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        # 更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2


def testing(dataset, labelset, weight1, weight2, value1, value2):
    # 记录预测正确的个数
    rightcount = 0
    out = []
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)  # 隐层输出
        output3 = sigmoid(np.dot(output2, weight2) - value2)   # 输出层输出
        # 输出预测结果
        # print("*************", output3)
        out.append(output3[0, 0])
    # 返回正确率
    return out


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
    dataset = np.zeros((int(N / 5) * 4 + 4, dim))
    test_dataset = np.zeros((int(N / 5), dim))
    for i in range(0, int(N / 5)):
        test_dataset[i] = input[5 * i]
        # 存放数据
        dataset[4 * i + 1] = (input[5 * i + 1])
        dataset[4 * i + 2] = (input[5 * i + 2])
        dataset[4 * i + 3] = (input[5 * i + 3])
        dataset[4 * i + 4] = (input[5 * i + 4])
    # 导入标签数据
    df2 = pd.read_excel('data4.xls', sheet_name='output-183', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    labelset = np.zeros(int(height / 5) * 4 + 4)
    test_labelset = np.zeros(int(height / 5))
    for i in range(0, int(height / 5)):
        test_labelset[i] = output[5 * i]
        # 存放数据
        labelset[4 * i + 1] = (output[5 * i + 1])
        labelset[4 * i + 2] = (output[5 * i + 2])
        labelset[4 * i + 3] = (output[5 * i + 3])
        labelset[4 * i + 4] = (output[5 * i + 4])
    weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), len(dataset[0]), 1)
    for i in range(1000):
        weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    test_out = testing(test_dataset, test_labelset, weight1, weight2, value1, value2)
    train_out = testing(dataset, labelset, weight1, weight2, value1, value2)
    test_err = np.sum((test_out - test_labelset) ** 2) / len(test_out)
    train_err = np.sum((train_out - labelset) ** 2) / len(train_out)
    end = time.perf_counter()  # 停止时间
    # 输出
    print('运行时间：%s' % (end - start))
    # 保存
    output_excel = pd.DataFrame(test_out)
    writer = pd.ExcelWriter('brnn.xls')
    output_excel.to_excel(writer, sheet_name='BPNN', startcol=0, index=False)
    writer.save()

    print('输出泛化误差为：', test_err)
    print(test_out)
    print(test_labelset)
    print('输出拟合误差为：', train_err)
    plt.plot(test_out)
    plt.show()