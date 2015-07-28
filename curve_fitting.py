import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b, c):
    return a * x**b * np.exp(-c/x)
f = 'alpha_en.txt'
data = np.loadtxt(f, delimiter=', ')
xdata = data[:,0]
ydata = data[:,1]
popt, pcov = curve_fit(func,xdata,ydata,p0=[1.0e4,0.5,15.0])
print popt
plt.plot(xdata,ydata,label='data')
plt.plot(xdata,func(xdata,popt[0],popt[1],popt[2]),label='fit')
plt.legend(loc=0)
plt.show()
