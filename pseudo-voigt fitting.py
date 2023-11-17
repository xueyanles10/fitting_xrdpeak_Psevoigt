import lmfit
import matplotlib.pyplot as plt
import numpy as np
from lmfit import report_fit

data = np.loadtxt('si.dat')
x = data[:, 0]
y = data[:, 1]

plt.scatter(x,y)

# define the function for fitting, the parameter of 1.0024396183602313 should be corrected or refined
def get_residual_Pse(pars ,x ,y):
    vals = pars.valuesdict()
    area_1 = vals['area_1']
    ceit_1 = vals['ceit_1']
    fwhm_1 = vals['fwhm_1']
    alpha_1 = vals['alpha_1']
    
    area_2 = area_1/2
    ceit_2 = vals['ceit_2']
    fwhm_2 = vals['fwhm_2']
    alpha_2 = vals['alpha_2']
    
    gaussian_1 = 1 / (fwhm_1 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_1 )/ fwhm_1 )** 2)
    lorenzian_1 = 1 / np.pi * (fwhm_1 / ((x - ceit_1) ** 2 + fwhm_1 ** 2))      
    pseudo_voigt_1 = (( 1 - alpha_1) * gaussian_1 + alpha_1 * lorenzian_1) * area_1
    
    gaussian_2 = 1 / (fwhm_2 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_2 )/ fwhm_2 )** 2)
    lorenzian_2 = 1 / np.pi * (fwhm_2 / ((x - ceit_2) ** 2 + fwhm_2 ** 2))
    pseudo_voigt_2 = (( 1 - alpha_2) * gaussian_2 + alpha_2 * lorenzian_1) * area_2
    
    pseudo_voigt = pseudo_voigt_1 + pseudo_voigt_2 
    
    
    return pseudo_voigt - y
fit_params = lmfit.create_params(area_1=1000, ceit_1=88, fwhm_1=0.2, alpha_1=0.5,ceit_2=88.2, fwhm_2=0.15, alpha_2=0.5)

out = lmfit.minimize(get_residual_Pse, fit_params, args=(x, y))

def pseudo_voigt_1_fun(ceit_1, fwhm_1, alpha_1, area_1):
    
    gaussian_1 = 1 / (fwhm_1 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_1 )/ fwhm_1 )** 2)
    lorenzian_1 = 1 / np.pi * (fwhm_1 / ((x - ceit_1) ** 2 + fwhm_1 ** 2))      
    pseudo_voigt_1 = (( 1 - alpha_1) * gaussian_1 + alpha_1 * lorenzian_1) * area_1
    
    return pseudo_voigt_1

def pseudo_voigt_2_fun(ceit_2, fwhm_2, alpha_2, area_2):
    
    gaussian_2 = 1 / (fwhm_2 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_2 )/ fwhm_2 )** 2)
    lorenzian_2 = 1 / np.pi * (fwhm_2 / ((x - ceit_2) ** 2 + fwhm_2 ** 2))
    pseudo_voigt_2 = (( 1 - alpha_2) * gaussian_2 + alpha_2 * lorenzian_2) * area_2
    
    return pseudo_voigt_2

def double_pseudo_voigt_fun(ceit_1, fwhm_1, alpha_1, area_1,ceit_2, fwhm_2, alpha_2, area_2):
    
    gaussian_1 = 1 / (fwhm_1 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_1 )/ fwhm_1 )** 2)
    lorenzian_1 = 1 / np.pi * (fwhm_1 / ((x - ceit_1) ** 2 + fwhm_1 ** 2))      
    pseudo_voigt_1 = (( 1 - alpha_1) * gaussian_1 + alpha_1 * lorenzian_1) * area_1
                               
    gaussian_2 = 1 / (fwhm_2 * np.sqrt( 2 *np.pi) ) *np.exp(-0.5 *(( x -ceit_2 )/ fwhm_2 )** 2)
    lorenzian_2 = 1 / np.pi * (fwhm_2 / ((x - ceit_2) ** 2 + fwhm_2 ** 2))
    pseudo_voigt_2 = (( 1 - alpha_2) * gaussian_2 + alpha_2 * lorenzian_1) * area_2
    
    double_pseudo_voigt = pseudo_voigt_1 +  pseudo_voigt_2
                
    return double_pseudo_voigt 


pseudo_voigt_1_value = pseudo_voigt_1_fun(88.0110306, 0.04091999, 0.79208529, 940.167795)
pseudo_voigt_2_value = pseudo_voigt_2_fun(88.2893814, 0.04458272, 0.16480580, 940.167795/2)
pseudo_double_voigt_value = pseudo_voigt_1_value + pseudo_voigt_2_value


plt.scatter(x,y)
plt.plot(x,pseudo_voigt_1_value,'r-')
plt.plot(x,pseudo_voigt_2_value,'g-')
plt.plot(x,pseudo_double_voigt_value,'b-')
