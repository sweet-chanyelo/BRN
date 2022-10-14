import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 导入数据
df = pd.DataFrame(pd.read_excel('data3.xls', sheet_name='input-183'))
# 导入标签数据
df1 = pd.read_excel('data3.xls', sheet_name='output-183')
df2 = pd.concat([df, df1], axis=1)  # 数据合并
print(df2.head())  # 查看数据结构类型
# 画图
plt.figure()
sns.heatmap(df2.corr(method='spearman'), annot=True, cmap='rainbow', linecolor='white', linewidths=0.1)
plt.ylabel('pearson correlation of feature', y=1.05, size=15)
plt.show()



# #
# # plt.title('' , y=1.05, size=15)
# # g = sns.stripplot(x='Feature', y='result', data=input, jitter=True)
# # plt.xticks(rotation=45)
# # colormap = plt.cm.RdBu
# #
# #
# # 画热力图
# # print(df)
# df1.head()
#
