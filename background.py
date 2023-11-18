import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.optimize import curve_fit

# 读取XRD数据
file_path = 'si_1.dat'  # 替换为你的文件路径
data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['angle', 'intensity'])

# 定义Chebyshev多项式函数
def chebyshev_background(x, *coeffs):
    return np.polynomial.chebyshev.chebval(x, coeffs)

# 初始化Chebyshev多项式的阶数
order = 6

# 提取XRD数据的角度和强度
x_data = data['angle']
y_data = data['intensity']

# 初始猜测参数
initial_guess = np.zeros(order + 1)

# 使用curve_fit进行拟合
coeffs, _ = curve_fit(chebyshev_background, x_data, y_data, p0=initial_guess)

# 生成拟合后的背景数据
background_fit = chebyshev_background(x_data, *coeffs)

# 绘制XRD图和拟合的背景
plt.plot(x_data, y_data, label='XRD Data')
plt.plot(x_data, background_fit, label='Chebyshev Background', linestyle='--')
plt.xlabel('XRD Angle')
plt.ylabel('Intensity')
plt.title('XRD Pattern with Chebyshev Background')
plt.legend()
plt.show()