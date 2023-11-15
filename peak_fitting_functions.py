


# First function_Voigt function, used to fit the diffraction peaks 
from scipy.special import wofz
def voigt(x, center, sigma, gamma, area):
    """
    计算Voigt函数的实部

    参数：
    x: 自变量
    center: Voigt函数的峰位置参数
    sigma: 高斯函数的标准差
    gamma: 洛伦兹函数的半宽度
    amplitude: Voigt函数的振幅参数

    返回：
    Voigt函数的实部在给定点x处的值
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    result = area * np.real(wofz(z) / (sigma * np.sqrt(2 * np.pi)))
    return result


# Second function_pseudo-function, used to fit the diffraction peaks,A is the area of the diffractin peaks;2θ_1 is is the diffraction peaks;
# η_1 is the ratio of the guassian and lorentzian function
def pseudo_voigt function(x,A,2θ,β,η)：
    gaussian_pse = 1 / (β * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ)/β_1) ** 2)
    lorentzian_pse = 1 / np.pi*(β/((x-2θ)**2+β**2))
    return  ((1 + η) * gaussian_pse + η * lorentzian_pse) * A




# Second function_pseudo-function, used to fit the diffraction peaks with kα1 和kα2
def double_Pse_Voi_fun(x,A_1,2θ_1,β_1,η_1,2θ_2,β_2,η_2):
    gaussian_pse_1 = 1 / (β_1 * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ_1)/β_1) ** 2)
    lorentzian_pse_1 = 1 / np.pi*(β_1/((x-2θ_1)**2 + β_1**2))
    
    voigt1 = ((1-η_1) * gaussian_pse_1 + η_1 * lorentzian_pse_1) * A_1
       
    A_2 = A_1/2
    gaussian_pse_1 = 1 / (β_2 * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ_2)/β_2) ** 2)
    lorentzian_pse_1 = 1 / np.pi*(β_2/((x-2θ_2)**2 + β_2**2))
    voigt2 = ((1-η_2) * gaussian_pse_2 + η_2 * lorentzian_pse_2) * A_2
    
    return (voigt1 + voigt2) * 2/3