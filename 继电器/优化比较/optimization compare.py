from scipy.interpolate import make_interp_spline as spline
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

errr0 = [0.07171620257846227, 0.057053049092685014, 0.0466465387754043, 0.04093470405451112, 0.040934704054511106,
         0.04093470405451111, 0.04093470405451111, 0.04093470405451111, 0.04093470405451111, 0.0409538204865632,
         0.0409538204865632, 0.04098411696931061, 0.040984116969310605, 0.040984116969310605, 0.040984116969310605,
         0.040984116969310605, 0.040984116969310605, 0.0409841169693106, 0.040984116969310605, 0.040984116969310605,
         0.04098411696931056]
errr1 = [0.08416080486849126, 0.1052130791027356, 0.1052130791027356, 0.09333454190567932, 0.07635616916942253,
         0.07635616916942249, 0.07635616916942249, 0.07633559977611622, 0.07633559977611622, 0.07633559977611622,
         0.07632955412144005, 0.07632687552920221, 0.07632687552920221, 0.0763268755292022, 0.07632323415447982,
         0.07632038003442274, 0.07632054763719724, 0.07632323415447984, 0.07632040760493095, 0.07632323415447984,
         0.07632323415447981]
errr2 = [0.05776706179882326, 0.05477154037636508, 0.03490659406424146, 0.02894334629681251, 0.028943346296812507,
         0.022944596310784403, 0.010698892987406382, 0.00962277227941949, 0.00939687076409004, 0.009396870764090029,
         0.009396870764090034, 0.009388689305589188, 0.00938868930558919, 0.009388689305589185, 0.009388689305589185,
         0.009388689305589185, 0.009388689305589185, 0.009388689305589178, 0.009388689305589185, 0.009388689305589164,
         0.009388689305589176]
errr3 = [0.04918022609328115, 0.04282263523449143, 0.013402664661792291, 0.010329163336129016, 0.009736125209393298,
         0.009655518117861774, 0.009592413966747421, 0.00958195766974553, 0.00957782398148133, 0.009573911213188329,
         0.009574131487292059, 0.009574133235512585, 0.009574052021799213, 0.009573966518583627, 0.009571846396260238,
         0.00957185219328564, 0.009572099774135857, 0.009570370155233078, 0.009570382125492896, 0.00957043165627872,
         0.0095704243160847]

x_smooth = np.linspace(x.min(), x.max(), 200)
y1_smooth = spline(x, errr0)(x_smooth)
y2_smooth = spline(x, errr1)(x_smooth)
y3_smooth = spline(x, errr2)(x_smooth)
y4_smooth = spline(x, errr3)(x_smooth)
plt.figure(figsize=(10, 4.5))
plt.plot(x_smooth, y1_smooth, 'b--', lw=1.5, label='The MS-MSE of BRN0')
plt.scatter(x, errr0, color='', marker='v', edgecolors='b', s=24)

plt.plot(x_smooth, y2_smooth, 'm--', lw=1.5, label='The MS-MSE of BRN1')
plt.scatter(x, errr1, color='', marker='s', edgecolors='b', s=24)

plt.plot(x_smooth, y3_smooth, 'g--', lw=1.5, label='The MS-MSE of BRN2')
plt.scatter(x, errr2, color='', marker='o', edgecolors='b', s=24)

plt.plot(x_smooth, y4_smooth, 'r--', lw=1.5, label='The MS-MSE of BRN3')
plt.scatter(x, errr3, color='', marker='*', edgecolors='b', s=26)
x0 = 5
y0 = 0.0135
x1 = 18
y1 = 0.023
plt.scatter(x0, y0, color='', marker='o', edgecolors='r', s=70)
plt.plot([5, 0.0135], [0.0135, 0.0135], 'k-.', lw=1)
plt.text(4, 0.009, r'$(5,0.0135)$', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': '9', 'color': 'r'})
# plt.scatter(x1, y1, color='', marker='o', edgecolors='r', s=70)
# plt.plot([18, 0.023], [0.023, 0.023], 'k-.', lw=1)
# plt.text(17, 0.019, r'$(18,0.023)$', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': '9', 'color': 'r'})
# 画布设置
plt.xlim(-0.1, 21)           # x幅度
plt.ylim(0, 0.12)           # y幅度
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置图例字体
font1 = {'family': 'Times New Roman', 'style': 'oblique', 'weight': 'normal', 'size': 12}  # 设置字体
m = range(0, 21, 1)
plt.xticks(m, family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.xlabel("Layers", font1)       # x坐标轴
plt.ylabel("MS-MSE ", font1)  # y坐标轴
# plt.title('health state')
plt.legend(prop=font)
plt.grid(axis='y', linestyle='-.')
plt.show()