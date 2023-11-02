import lmfit
import matplotlib.pyplot as plt
import numpy as np
from lmfit import report_fit

data = np.loadtxt('si.dat')
x = data[:, 0]
y = data[:, 1]

plt.scatter(x,y)

# define the function for fitting
def get_residual_Pse(pars ,x ,y):
    vals = pars.valuesdict()
    area_1 = vals['area_1']
    ceit_1 = vals['ceit_1']
    sigma_1 = vals['sigma_1']
    alpha_1 = vals['alpha_1']
    
    area_2 = vals['area_2']
    ceit_2 = np.arcsin(1.0024396183602313 * np.sin(ceit_1/2*np.pi/180))*360/np.pi
    sigma_2 = vals['sigma_2']
    alpha_2 = vals['alpha_2']
        

    model = ( 1 -alpha_1) * area_1 /(sigma_1 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_1 )/ sigma_1 )** 2) + alpha_1 * area_1 / np.pi * (sigma_1 / ((x - ceit_1) ** 2 + sigma_1 ** 2))+( 1 -alpha_2) * area_2 /(sigma_2 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_2 )/ sigma_2 )** 2) + alpha_2 * area_2 / np.pi * (sigma_2 / ((x - ceit_2) ** 2 + sigma_2 ** 2))
    return model - y

fit_params = lmfit.create_params(area_1=1000, ceit_1=88, sigma_1=0.2, alpha_1=0.5,area_2=500, sigma_2=0.15, alpha_2=0.5)

out = lmfit.minimize(get_residual_Pse, fit_params, args=(x, y))

def double_Pse_Voi_fun(x,A_1,μ_1,σ_1,η_1,A_2,σ_2,η_2):
    μ_2 = np.arcsin(1.0024396183602313 * np.sin(μ_1/2*np.pi/180))*360/np.pi
    return (1-η_1) * A_1/(σ_1*np.sqrt(2*np.pi))*np.exp(-0.5*((x-μ_1)/σ_1)**2) + η_1 * A_1/np.pi*(σ_1/((x-μ_1)**2+σ_1**2))+(1-η_2) * A_2/(σ_2*np.sqrt(2*np.pi))*np.exp(-0.5*((x-μ_2)/σ_2)**2) + η_2 * A_2/np.pi*(σ_2/((x-μ_2)**2+σ_2**2))

voigt_init = double_Pse_Voi_fun(x,1000, 88, 0.2, 0.5,500, 0.15, 0.5)
voigt_best_fit = double_Pse_Voi_fun(x,947.850357,88.0119455,0.04007277,0.80580580,490.817701,0.04593350, 0.69306092)

plt.scatter(x,y)
# plt.plot(x,voigt_init, 'r-')
plt.plot(x,voigt_best_fit, 'g-')
plt.show()