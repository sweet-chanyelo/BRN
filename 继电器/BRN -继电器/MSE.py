from scipy.interpolate import make_interp_spline as spline
import matplotlib.pyplot as plt
import numpy as np
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
errr1 = [0.07260783807641866, 0.03175406594827769, 0.02680697410492507, 0.020673243073022297, 0.019032026220760667,
         0.013091436813544594, 0.013926662228149478, 0.012929098553109636, 0.012260056635614865, 0.012754474716790342,
         0.013404512016230125, 0.012154337408674156, 0.01284571130024086, 0.012666638110569431, 0.012999384210210352,
         0.012795701921953829, 0.011931594988078326, 0.012246378315455216, 0.011679050962386504, 0.01191274328510919,
         0.012989159403429434]
errr0 = [0.08182566654327493, 0.053367869350782596, 0.06159740909323062, 0.03763660446866643, 0.04330583753138166,
         0.040493213039969486, 0.03377812429887006, 0.04955281004182814, 0.04712739444733014, 0.03354983149763804,
         0.04229161350701978, 0.025290318681433506, 0.02378546876762518, 0.026992598340206436, 0.024763938829638787,
         0.03155917597975918, 0.02544179765847641, 0.03176094866422658, 0.023311120348998666, 0.02626447160716645,
         0.024761171898285341]
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = spline(x, errr1)(x_smooth)
z_smooth = spline(x, errr0)(x_smooth)
plt.figure(figsize=(10, 4.5))
plt.plot(x_smooth, y_smooth, 'b--', lw=1.5, label='The MSE of BRN1')
plt.scatter(x, errr1, color='', marker='v', edgecolors='b', s=24)
plt.plot(x_smooth, z_smooth, 'g--', lw=1.5, label='The MSE of BRN0')
plt.scatter(x, errr0, color='', marker='s', edgecolors='g', s=23)
x0 = 5
y0 = 0.0135
x1 = 18
y1 = 0.023
plt.scatter(x0, y0, color='', marker='o', edgecolors='r', s=70)
plt.plot([5, 0.0135], [0.0135, 0.0135], 'k-.', lw=1)
plt.text(4, 0.009, r'$(5,0.0135)$', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': '9', 'color': 'r'})
plt.scatter(x1, y1, color='', marker='o', edgecolors='r', s=70)
plt.plot([18, 0.023], [0.023, 0.023], 'k-.', lw=1)
plt.text(17, 0.019, r'$(18,0.023)$', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': '9', 'color': 'r'})
# 画布设置
plt.xlim(-0.1, 21)           # x幅度
plt.ylim(0, 0.09)           # y幅度
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置图例字体
font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'normal', 'size': 12}  # 设置字体
m = range(0, 21, 1)
plt.xticks(m, family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.xlabel("Layers", font1)       # x坐标轴
plt.ylabel("MSE ", font1)  # y坐标轴
# plt.title('health state')
plt.legend(prop=font)
plt.grid(axis='y', linestyle='-.')
plt.show()