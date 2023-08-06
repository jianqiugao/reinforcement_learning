
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

# 定义x轴上的点
x = np.linspace(0, 1, 100)

# 定义不同参数下的beta分布
a1, b1 = 6,8 # 均匀分布
a2, b2 = 4, 12 # 正偏态分布
a3, b3 = 12, 4 # 负偏态分布
a4, b4 = 2, 2 # 对称分布
a5, b5 = 5, 5 # 双峰分布
res = np.random.beta(a2,b2,1000) # 每次都不一样
print(res)
# 计算不同参数下的概率密度函数值
y1 = beta.pdf(x, a1, b1)
y2 = beta.pdf(x, a2, b2)
y3 = beta.pdf(x, a3, b3)
# y4 = beta.pdf(x, a4, b4)
# y5 = beta.pdf(x, a5, b5)

# 绘制不同参数下的概率密度函数图像
plt.plot(x, y1, label="Beta(6,8)")
plt.plot(x, y2, label="Beta(4,12)")
plt.plot(x, y3, label="Beta(12,4)")
for x in res:
    plt.axvline(x)
# plt.plot(x, y4, label="Beta(2,2)")
# plt.plot(x, y5, label="Beta(2,5)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Beta distribution")
plt.legend()
plt.show()
