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