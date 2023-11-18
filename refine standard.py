def calculate_rwp(observed_intensity, calculated_intensity, weights):
    numerator = np.sqrt(np.sum(weights * (observed_intensity - calculated_intensity)**2))
    denominator = np.sum(weights * observed_intensity**2)
    rwp = numerator / np.sqrt(denominator)
    return rwp

def calculate_rp(observed_intensity, calculated_intensity):
    numerator = np.sqrt(np.sum((observed_intensity - calculated_intensity)**2))
    denominator = np.sqrt(np.sum(observed_intensity**2))
    rp = numerator / denominator
    return rp

def calculate_chi_square(observed_intensity, calculated_intensity, errors):
    chi_square = np.sum(((observed_intensity - calculated_intensity) / errors)**2)
    return chi_square

def calculate_rexp(observed_intensity, calculated_intensity):
    residual = observed_intensity - calculated_intensity
    rexp = np.sum(np.abs(residual)) / np.sum(observed_intensity)
    return rexp