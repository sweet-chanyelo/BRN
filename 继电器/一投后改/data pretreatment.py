"""
实验共12组数据，其中一组为常值，为此程序中没有体现，只有11个变量
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    df1 = pd.read_excel('input.xlsx', 'data')
    row, col = df1.shape
    # print(df1)
    point = range(0, 65000, 300)
    print(point)
    data = np.zeros((len(point), col))
    for i in range(len(point)):
        data[i] = df1.iloc[point[i]]
        data[i][1] = data[i][1] / 1000
    print(data)
    plt.plot(data[:, 4])
    plt.show()
    # 输出
    output_excel = pd.DataFrame(data)
    writer = pd.ExcelWriter('input2.xlsx')
    output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    writer.save()
