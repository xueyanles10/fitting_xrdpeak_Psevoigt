###
The Gaussian function is possibly the best-known peak function in the whole of science since many physical and chemical processes are governed by Gaussian statistics. Translated into powder diffraction terms, the function for the intensity at any value of 2θ near the peak becomes:
where Imax is the peak intensity, 2θ0 is the 2θ position of the peak maximum, and the integral breadth, β, is related to the FWHM peak width, H, by β = 0.5 H (π / loge2)1/2. The most important features of the Gaussian function are:

that it is easy to calculate
it is a familiar and well-understood function
it is a good function to describe both neutron and energy-dispersive X-ray powder diffraction peaks (it is however not good at describing angle-dispersive X-ray diffraction peaks)
it has a convenient convolution property (see later)
it is symmetrical
###

def gaussian(x,2θ,β,A):
    gaussian_fun = A / (β * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ)/β_1) ** 2)

###
lorentzaian function
where w is equal to half of the peak width (w = 0.5 H). The main features of the Lorentzian function are:
that it is also easy to calculate
that, relative to the Gaussian function, it emphasises the tails of the peak
its integral breadth β = π H / 2
it has a convenient convolution property 
it is symmetrical
instrumental peak shapes are not normally Lorentzian except at high angles where wavelength dispersion is dominan
###

def lorentzian(x,2θ,β,A):
    lorentzian_fun = A / np.pi*(β/((x-2θ)**2+β**2))
    return lorentzian_fun



# Voigt function, used to fit the diffraction peaks 
from scipy.special import wofz
def voigt(x, 2θ, βg, βl, A):
    """
    计算Voigt函数的实部

    参数：
    x: 自变量
    center: Voigt函数的峰位置参数
    sigma βg: 高斯函数的标准差
    gamma βl: 洛伦兹函数的半宽度
    amplitude: Voigt函数的振幅参数

    返回：
    Voigt函数的实部在给定点x处的值
    """
    z = ((x - 2θ) + 1j * βl) / (βg * np.sqrt(2))
    voigt_fun = A * np.real(wofz(z) / (βg * np.sqrt(2 * np.pi)))
    return voigt_fun

###
These combine different functions in an attempt to get the "best of both worlds" as far as peak shape is concerned. The combination can be by convolution (e.g. the Voigt function) or by simple addition (e.g. pseudo-Voigt which is a close approximation to the Voigt function). For example the latter case could take the form:
pseudo-function, used to fit the diffraction peaks,A is the area of the diffractin peaks;2θ_1 is is the diffraction peaks;
η_1 is the ratio of the guassian and lorentzian function
η (the "Lorentz fraction") and (1 − η) represent the fractions of each used. The main features of combination functions are:
that their precise form of combination can be tailored to a specific peak shape
The Voigt and pseudo-Voigt (together with the Pearson VII) are popular functions for modelling peak shapes
that their calculation tends to be more labour-intensive though as stated before this is not significant in computing terms
###
 
def pseudo_voigt function(x,A,2θ,β,η)：
    gaussian_pse = 1 / (β * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ)/β_1) ** 2)
    lorentzian_pse = 1 / np.pi*(β/((x-2θ)**2+β**2))
    pseudo_voigt_fun = ((1 + η) * gaussian_pse + η * lorentzian_pse) * A
    
    return  pseudo_voigt_fun




# double function_pseudo-function, used to fit the diffraction peaks with kα1 和kα2
def double_Pseudo_Voigt(x,A_1,2θ_1,β_1,η_1,2θ_2,β_2,η_2):
    gaussian_pse_1 = 1 / (β_1 * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ_1)/β_1) ** 2)
    lorentzian_pse_1 = 1 / np.pi*(β_1/((x-2θ_1)**2 + β_1**2))
    voigt1 = ((1-η_1) * gaussian_pse_1 + η_1 * lorentzian_pse_1) * A_1
       
    A_2 = A_1/2
    gaussian_pse_1 = 1 / (β_2 * np.sqrt(2*np.pi))*np.exp(-0.5 *((x - 2θ_2)/β_2) ** 2)
    lorentzian_pse_1 = 1 / np.pi*(β_2/((x-2θ_2)**2 + β_2**2))
    voigt2 = ((1-η_2) * gaussian_pse_2 + η_2 * lorentzian_pse_2) * A_2
        
    double_pseudo_fun = (voigt1 + voigt2) * 2/3
    return double_pseudo_fun
    

    


###
The Pearson VII function was a popular function during the 1980s and 1990s for describing peak shapes from conventional X-ray powder diffraction patterns, though it has now been superceded in popularity by the pseudo-Voigt peak-shape function (described on the next page). The Pearson VII function is basically a Lorentz function raised to a power m,I is the intensity, where m can be chosen to suit a particular peak shape and w is related to the peak width. Special cases of this function are that it becomes a Lorentzian as m → 1 and approaches a Gaussian as m → ∞ (e.g. m > 10). The main features of the Lorentzian function are:that it can handle different tail-shapes of a peak, better than say a Gaussian or Lorentzian function, by varying the m parameter its calculation is simpler than some of its competitors, though this is not significant in computing terms
###
def pearson_VII_fun(x,2θ,I,m,β):
    pearson_fun = I * β ** (2 * m) / (β**2 + 2 ** (1/m)*(x-2θ)**2) ** m
    return pearson_VII_fun

#The Finger–Cox–Jephcoat (FCJ) function is used to describe the pair correlation function of a fluid in statistical mechanics. It is commonly used in the context of molecular dynamics simulations. 
def fcj_function(r, rho, a, b, c):
    """
    Finger–Cox–Jephcoat (FCJ) pair correlation function.

    Parameters:
    - r: Array of radial distances.
    - rho: Number density of the fluid.
    - a, b, c: Parameters of the FCJ function.

    Returns:
    - g(r): Pair correlation function values at each radial distance.
    """
    g_r = np.exp(-a*r) + b*np.exp(-c*r**2)
    g_r *= np.exp(rho*(a + 2*c*r) / (1 + 2*c*r))
    return g_r
