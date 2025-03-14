import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

real_f = open("real.txt", "r")
pred_f = open("pred.txt", "r")

real = real_f.readlines()
pred = pred_f.readlines()
real = list(map(lambda x: float(x.strip()), real))
pred = list(map(lambda x: float(x.strip()), pred))
pass

minus = [real[i] - pred[i] for i in range(len(real))]
# minus = real

# 生成区间
sections = np.arange(-3, 3, 0.1)
# sections = np.arange(0, 7, 0.1)

# 画出直方图
plt.hist(minus, bins=sections, density=True, alpha=0.6, color='b', label="Histogram")

# 画出概率密度估计（Kernel Density Estimation）

kde = stats.gaussian_kde(minus)
x_vals = np.linspace(-3, 3, 300)  # 生成密度曲线的 x 值
# x_vals = np.linspace(0, 7, 300)
plt.plot(x_vals, kde(x_vals), color='r', label="KDE")

# 添加标签和标题
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Estimated Distribution")
plt.legend()

# 显示图像
plt.show()

# 绘制 Q-Q 图
plt.figure(figsize=(6, 6))
stats.probplot(minus, dist="norm", plot=plt)

# 添加标题
plt.title("Q-Q Plot (Normal Distribution)")

# 显示图像
plt.show()
