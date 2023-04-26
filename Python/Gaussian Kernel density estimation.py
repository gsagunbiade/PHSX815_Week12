# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:23:26 2023

@author: Gbenga Agunbiade
"""

import numpy as np

np.random.seed(42)
data = np.random.normal(0, 1, size=100)
def gaussian_kernel_density(x, data, h):
    """
    Computes the Gaussian Kernel Density Estimate for a given dataset.
    
    Parameters:
    x: float
        The value at which to evaluate the density estimate.
    data: numpy.ndarray
        A one-dimensional numpy array containing the dataset.
    h: float
        The bandwidth parameter.
    
    Returns:
    The value of the density estimate at x.
    """
    n = data.shape[0]
    k = lambda u: np.exp(-0.5*u**2)/np.sqrt(2*np.pi)
    return np.sum(k((x-data)/h))/(n*h)
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 1000)

hs = [0.1, 0.5, 1, 1.5, 2, 2.5, 3]
densities = []

for h in hs:
    density = []
    for xi in x:
        density.append(gaussian_kernel_density(xi, data, h))
    densities.append(density)
plt.hist(data, bins=20, density=True, alpha=0.5, label='Data')

for i, density in enumerate(densities):
    plt.plot(x, density, label='h='+str(hs[i]))

plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gaussian Kernel Density Estimation')
plt.show()
