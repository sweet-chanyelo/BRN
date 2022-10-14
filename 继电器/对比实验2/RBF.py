"""

"""
import numpy as np
from math import sqrt
import pandas as pd


def load_data():
    df1 = pd.read_excel('data3.xls', sheet_name='input-183', header=None)
    # 样本量与数据维度
    N, dim = df1.shape
    input = np.zeros((N - 1, dim))
    for i in range(1, N):
        for j in range(0, dim):
            input[i - 1][j] = df1.iat[i, j]
    # print(input)
    df2 = pd.read_excel('data3.xls', sheet_name='output-183', header=None)
    height, width = df2.shape
    output = np.zeros(height - 1)  # 默认为只有1列（单输出）
    for i in range(1, height):
        output[i - 1] = df2.iat[i, 0]
    # print(output)
    n_output = 1

    return np.mat(input), np.mat(output), n_output


def linear(x):
    '''Sigmoid函数（输出层神经元激活函数）
    input:  x(mat/float):自变量，可以是矩阵或者是任意实数
    output: Sigmoid值(mat/float):Sigmoid函数的值
    '''
    return x


def hidden_out(feature, center, delta):
    '''rbf函数（隐含层神经元输出函数）
    input:feature(mat):数据特征
          center(mat):rbf函数中心
          delta(mat)：rbf函数扩展常数
    output：hidden_output（mat）隐含层输出
    '''
    m, n = feature.shape
    m1, n1 = center.shape
    hidden_out = np.mat(np.zeros((m, m1)))
    for i in range(m):
        for j in range(m1):
            hidden_out[i, j] = np.exp(-1.0 * (feature[i, :] - center[j, :]) * (feature[i, :] - center[j, :]).T / (
                        2 * delta[0, j] * delta[0, j]))
    return hidden_out


def predict_in(hidden_out, w):
    '''计算输出层的输入
    input:  hidden_out(mat):隐含层的输出
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: predict_in(mat):输出层的输入
    '''
    m = hidden_out.shape[0]
    predict_in = hidden_out * w
    return predict_in


def predict_out(predict_in):
    '''输出层的输出
    input:  predict_in(mat):输出层的输入
    output: result(mat):输出层的输出
    '''
    result = linear(predict_in)
    return result


def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    '''计算隐含层的输入
    input:  feature(mat):特征
            label(mat):标签
            n_hidden(int):隐含层的节点个数
            maxCycle(int):最大的迭代次数
            alpha(float):学习率
            n_output(int):输出层的节点个数
    output: center(mat):rbf函数中心
            delta(mat):rbf函数扩展常数
            w(mat):隐含层到输出层之间的权重
    '''
    m, n = feature.shape
    # 1、初始化
    center = np.mat(np.random.rand(n_hidden, n))
    center = center * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - np.mat(np.ones((n_hidden, n))) * (
                4.0 * sqrt(6) / sqrt(n + n_hidden))
    delta = np.mat(np.random.rand(1, n_hidden))
    delta = delta * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - np.mat(np.ones((1, n_hidden))) * (
                4.0 * sqrt(6) / sqrt(n + n_hidden))
    w = np.mat(np.random.rand(n_hidden, n_output))
    w = w * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - np.mat(np.ones((n_hidden, n_output))) * (
                4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    # 2、训练
    iter = 0
    while iter <= maxCycle:
        # 2.1、信号正向传播
        # 2.1.1、计算隐含层的输出
        hidden_output = hidden_out(feature, center, delta)
        # 2.1.3、计算输出层的输入
        output_in = predict_in(hidden_output, w)
        # 2.1.4、计算输出层的输出
        output_out = predict_out(output_in)

        # 2.2、误差的反向传播
        error = np.mat(label - output_out)
        print(label.shape, output_out.shape, error.shape)
        for j in range(n_hidden):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            for i in range(m):
                a = np.exp(-1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j]))
                print(a.shape, error[i, :].shape, (feature[i] - center[j]).shape)
                sum1 += error[i, :] * a * (feature[i] - center[j])
                sum2 += error[i, :] * np.exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    feature[i] - center[j]) * (feature[i] - center[j]).T
                sum3 += error[i, :] * np.exp(
                    -1.0 * (feature[i] - center[j]) * (feature[i] - center[j]).T / (2 * delta[0, j] * delta[0, j]))
            delta_center = (w[j, :] / (delta[0, j] * delta[0, j])) * sum1
            delta_delta = (w[j, :] / (delta[0, j] * delta[0, j] * delta[0, j])) * sum2
            delta_w = sum3
            # 2.3、 修正权重和rbf函数中心和扩展常数
            center[j, :] = center[j, :] + alpha * delta_center
            delta[0, j] = delta[0, j] + alpha * delta_delta
            w[j, :] = w[j, :] + alpha * delta_w
        if iter % 10 == 0:
            cost = (1.0 / 2) * get_cost(get_predict(feature, center, delta, w) - label)
            print("\t-------- iter: ", iter, " ,cost: ", cost)
        if cost < 3:  ###如果损失函数值小于3则停止迭
            break
        iter += 1
    return center, delta, w


def get_cost(cost):
    '''计算当前损失函数的值
    input:  cost(mat):预测值与标签之间的差
    output: cost_sum / m (double):损失函数的值
    '''
    m, n = cost.shape

    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i, j] * cost[i, j]
    return cost_sum / m


def get_predict(feature, center, delta, w):
    '''计算最终的预测
    input:  feature(mat):特征
            w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: 预测值
    '''
    return predict_out(predict_in(hidden_out(feature, center, delta), w))


def save_model_result(center, delta, w, result):
    '''保存最终的模型
    input:  w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output:
    '''

    def write_file(file_name, source):
        f = open(file_name, "w")
        m, n = source.shape
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
        f.close()

    write_file("center.txt", center)
    write_file("delta.txt", delta)
    write_file("weight.txt", w)
    write_file('train_result.txt', result)


def err_rate(label, pre):
    '''计算训练样本上的错误率
    input:  label(mat):训练样本的标签
            pre(mat):训练样本的预测值
    output: rate[0,0](float):错误率
    '''

    m = label.shape[0]
    for j in range(m):
        if pre[j, 0] > 0.5:
            pre[j, 0] = 1.0
        else:
            pre[j, 0] = 0.0

    err = 0.0
    for i in range(m):
        if float(label[i, 0]) != float(pre[i, 0]):
            err += 1
    rate = err / m
    return rate


if __name__ == "__main__":
    # 1、导入数据
    print("--------- 1.load data ------------")
    feature, label, n_output = load_data()
    # 2、训练网络模型
    print("--------- 2.training ------------")
    center, delta, w = bp_train(feature, label, 20, 5000, 0.008, n_output)
    # 3、得到最终的预测结果
    print("--------- 3.get prediction ------------")
    result = get_predict(feature, center, delta, w)
    print("训练准确性为：", (1 - err_rate(label, result)))
    # 4、保存最终的模型
    print("--------- 4.save model and result ------------")
    save_model_result(center, delta, w, result)