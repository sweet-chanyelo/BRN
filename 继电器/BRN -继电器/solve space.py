import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
x3 = sp.Symbol('x3')
a1 = sp.Symbol('a1')
a2 = sp.Symbol('a2')
a3 = sp.Symbol('a3')
b1 = sp.Symbol('b1')
b2 = sp.Symbol('b2')
b3 = sp.Symbol('b3')
c1 = sp.Symbol('c1')
c2 = sp.Symbol('c2')
c3 = sp.Symbol('c3')

# fx1 =
# fx2 = x2 - 0.7
# fx3 = x3 - 0.5

f1 = 1/12 * x1 + 1/3 * x2 + 5/24 * x3 - 7/18 * x1 * x2 - 7/24 * x1 * x2 - 17/36 * x1 * x3 + 19/36 * x1 * x2 * x3



print(sp.solve([f1], [x1, x2, x3]))
figure = plt.figure(figsize=(6.4, 5))
ax = Axes3D(figure)
X = np.arange(0.15, 1, 0.05)  # x3
Y = np.arange(0.15, 1, 0.05)
X, Y = np.meshgrid(X, Y)
Z = -(24 * X + 15 * Y)/(38 * X * Y - 49 * X - 34 * Y + 6)
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 13}  # 设置坐标轴字体
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  # 设置字体
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_zlim(0, 1)
ax.set_xlabel("rule unit weight $\Theta$2", font)
ax.set_ylabel("rule unit weight $\Theta$3", font)
ax.set_zlabel("rule unit weight $\Theta$1", font)
plt.grid(axis='both', linestyle='-.')
plt.show()
