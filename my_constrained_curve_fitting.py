import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit, minimize, leastsq
from scipy.integrate import quad
from scipy import pi, sin, exp, log

xdata = mean_energy
ydata = kArIz

for i in xrange(len(ydata) - 1, -1, -1):
    element = ydata[i]
    if element < 1e-30:
        # ydata = np.delete(ydata,i)
        # xdata = np.delete(xdata,i)
        ydata[i] = 0.

def lin_poly_func(x, a0, a1, a2,a3,a4,a5):
    y = a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
    return y

def log_poly_func(x, a0, a1, a2,a3,a4,a5):
    y = a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
    return y

def lin_penalty_func(x,p):
    model = lin_poly_func(x,p[0],p[1],p[2],p[3],p[4],p[5])
    modmin = np.min(model)
    if modmin < 0:
        model = model - modmin
    return model

# def Lieberman_func(x, a):
#     y = a*x*exp(-15.76/x)
#     return y
# def Lieberman_func(x, a, b):
#     y = a*x**b*exp(-15.76/x)
#     return y
def Lieberman_func(x, p):
    y = p[0]*x**p[1]*exp(p[2]*x**p[3]) #+ p[4]*x**p[5]*exp(-p[6]*x**p[7])
    return y
# def Lieberman_func(x, a, b, c):
#     y = a*x**b*exp(-c/x)
#     return y

def stupid_func(x, a, c):
    y = a*exp(-c/x)
    return y

def residuals(p,x,y,i):
    if i == 0:
        model = lin_penalty_func(x,p)
    elif i == 1:
        model = Lieberman_func(x,p)
    else:
        model = stupid_func(x,p[0],p[1])
    # print y
    # print model
    return y - model

figLin = plt.figure()
# figLog = plt.figure()
axLin = figLin.add_subplot(111)
# axLog = figLog.add_subplot(111)

axLin.scatter(xdata,ydata, marker='.',label='Data from Boltzmann solver')
# axLog.scatter(log_xdata,log_ydata, marker='.',label='bolos')

# popt_lin, pcov_lin = curve_fit(lin_poly_func,xdata,ydata,p0=[1.e-14,1.e-14,1.e-14,1.e-14,1.e-14,1.e-14])
# popt_log, pcov_log = curve_fit(log_poly_func,log_xdata,log_ydata,p0=[1.,1.,1.,1.,1.,1.])
# y_fit_lin = lin_poly_func(xdata, *popt_lin)
# y_fit_log = log_poly_func(log_xdata, *popt_log)
# axLin.plot(xdata,y_fit_lin, label='lin_curve_fit')
# axLin.plot(xdata,exp(y_fit_log),  label='log_curve_fit')
# axLog.plot(log_xdata,y_fit_log, label='log_curve_fit')
# axLog.plot(log_xdata,log(y_fit_lin), label='lin_curve_fit')

# popt_penalty, pcov_penalty = leastsq(func=residuals,x0=(1.,1.,1.,1.,1.,1.), args=(xdata,ydata,0))
# y_lin_penalty_fit = lin_penalty_func(xdata, popt_penalty)
# axLin.plot(xdata,y_lin_penalty_fit, label='constrained')

popt_exp, pcov_exp = leastsq(func=residuals,x0=(1e-15,1.,-17.,-1.), args=(xdata,ydata,1))
print popt_exp
y_exp_fit = Lieberman_func(xdata, popt_exp)
axLin.plot(xdata,y_exp_fit, label='Energy model fit')

# popt_stupid, pcov_stupid = leastsq(func=residuals,x0=(1e4,76.), args=(xdata,ydata,2))
# y_stupid_fit = stupid_func(xdata, *popt_stupid)
# axLin.plot(xdata,y_stupid_fit, label='stupid')

axLin.legend(loc=0)
axLin.set_ylim(1e-30,np.max(ydata))
axLin.set_xlim((np.min(xdata),np.max(xdata)))
axLin.set_yscale('log')
# axLin.set_xscale('log')
ticks = []
ticklabels = []
for i in range(1,10):
    ticks.append(i)
    ticklabels.append(str(i)) 
axLin.set_xticks(ticks)
axLin.set_xticklabels(ticklabels)
axLin.set_xlabel("Mean energy (eV)")
axLin.set_ylabel("Ionization rate coefficient (m$^3$/s)")
# axLog.legend(loc=0)
plt.show()

# print popt1
# print popt2
