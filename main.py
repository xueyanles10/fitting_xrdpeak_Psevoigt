import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 生成模拟的X射线衍射数据
def simulate_xrd_data(x, peak_positions, peak_intensities, background_level=0.1, noise_level=0.05):
    background = background_level * np.random.normal(size=len(x))
    peaks = sum([gaussian(x, intensity, position, 0.1) for intensity, position in zip(peak_intensities, peak_positions)])
    noise = noise_level * np.random.normal(size=len(x))
    return background + peaks + noise

# 定义高斯峰函数作为拟合模型
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

# 定义RietveldRefinement类
class RietveldRefinement:
    def __init__(self, x_data, observed_data, num_peaks):
        self.x_data = x_data
        self.observed_data = observed_data
        self.num_peaks = num_peaks
        self.background_level = 0.1
        self.peak_intensities = np.ones(num_peaks)
        self.peak_positions = np.linspace(min(x_data), max(x_data), num_peaks)
        self.scale_factor = 1.0

    def model_function(self, x, params):
        background = params[0]
        peak_params = params[1:].reshape((3, -1))
        peaks = sum([gaussian(x, intensity, position, width) for intensity, position, width in peak_params.T])
        return background + self.scale_factor * peaks

    def objective_function(self, params):
        fit_curve = self.model_function(self.x_data, params)
        residuals = fit_curve - self.observed_data
        return np.sum(residuals**2)

    def perform_refinement(self):
        initial_params = np.concatenate([[self.background_level], self.peak_intensities, self.peak_positions, np.ones(self.num_peaks)*0.1])
        result = minimize(self.objective_function, initial_params, method='L-BFGS-B')
        refined_params = result.x
        self.background_level = refined_params[0]
        self.peak_intensities = refined_params[1:self.num_peaks+1]
        self.peak_positions = refined_params[self.num_peaks+1:2*self.num_peaks+1]
        self.scale_factor = refined_params[-self.num_peaks:].mean()

    def plot_results(self):
        fit_curve = self.model_function(self.x_data, np.concatenate([[self.background_level], self.peak_intensities, self.peak_positions, np.ones(self.num_peaks)*0.1]))
        plt.plot(self.x_data, self.observed_data, label='Observed Data', linestyle='-', marker='o')
        plt.plot(self.x_data, fit_curve, label='Fit Curve', linestyle='--')
        plt.legend()
        plt.show()

# 生成模拟的X射线衍射数据
x_data = np.linspace(0, 10, 100)
peak_positions = [2, 5, 8]
peak_intensities = [5, 8, 4]
observed_data = simulate_xrd_data(x_data, peak_positions, peak_intensities)

# 创建RietveldRefinement实例
rietveld_refinement = RietveldRefinement(x_data, observed_data, num_peaks=3)

# 执行Rietveld精修
rietveld_refinement.perform_refinement()

# 绘制拟合结果
rietveld_refinement.plot_results()
