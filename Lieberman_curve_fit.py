import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit, minimize, leastsq
from scipy.integrate import quad
from scipy import pi, sin, exp, log

xdata = EArray
ydata = alpha

for i in xrange(len(ydata) - 1, -1, -1):
    element = ydata[i]
    if element < 1e-15:
        ydata = np.delete(ydata,i)
        xdata = np.delete(xdata,i)
log_xdata = log(xdata)
log_ydata = log(ydata)

def func(x, a, b, c):
    y = a*x**b*exp(-c/x)
    for i in range(0,len(y)):
        if y[i] < 0:
            y[i] = 0
    return y

def log_poly_func(x, a0, a1, a2,a3,a4,a5):
    y = a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
    return y

def residuals(p,x,y):
    model = lin_poly_func(x,p[0],p[1],p[2],p[3],p[4],p[5])
    modmin = np.min(model)
    if modmin < 0:
        model = model - modmin
    print model
    print y
    print model.shape
    print y.shape
    return y - model

figLin = plt.figure()
figLog = plt.figure()
axLin = figLin.add_subplot(111)
axLog = figLog.add_subplot(111)

axLin.scatter(xdata,ydata, marker='.',label='bolos')
axLog.scatter(log_xdata,log_ydata, marker='.',label='bolos')

popt_lin, pcov_lin = curve_fit(lin_poly_func,xdata,ydata,p0=[1.e-14,1.e-14,1.e-14,1.e-14,1.e-14,1.e-14])
popt_log, pcov_log = curve_fit(log_poly_func,log_xdata,log_ydata,p0=[1.,1.,1.,1.,1.,1.])
y_fit_lin = lin_poly_func(xdata, *popt_lin)
y_fit_log = log_poly_func(log_xdata, *popt_log)
axLin.plot(xdata,y_fit_lin, label='lin_curve_fit')
axLin.plot(xdata,exp(y_fit_log),  label='log_curve_fit')
axLog.plot(log_xdata,y_fit_log, label='log_curve_fit')
axLog.plot(log_xdata,log(y_fit_lin), label='lin_curve_fit')

popt_penalty, pcov_penalty = leastsq(func=residuals,x0=(1.e-14,1.e-14,1.e-14,1.e-14,1.e-14,1.e-14), args=(xdata,ydata))
y_lin_penalty_fit = lin_poly_func(xdata, *popt_penalty)
axLin.plot(xdata,y_lin_penalty_fit, label='constrained')

axLin.legend(loc=0)
axLog.legend(loc=0)
plt.show()

# print popt1
# print popt2
