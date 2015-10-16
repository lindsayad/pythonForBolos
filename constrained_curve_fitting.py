import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit, minimize, leastsq
from scipy.integrate import quad
from scipy import pi, sin
x = scipy.linspace(0, pi, 100)
y = scipy.sin(x) + (0. + scipy.rand(len(x))*0.4)
def func1(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

# here you include the penalization factor
def residuals(p,x,y):
    integral = quad( func1, 0, pi, args=(p[0],p[1],p[2],p[3]))[0]
    penalization = abs(2.-integral)*10000
    return y - func1(x, p[0],p[1],p[2],p[3]) - penalization

popt1, pcov1 = curve_fit( func1, x, y )
popt2, pcov2 = leastsq(func=residuals, x0=(1.,1.,1.,1.), args=(x,y))
y_fit1 = func1(x, *popt1)
y_fit2 = func1(x, *popt2)
plt.scatter(x,y, marker='.')
plt.plot(x,y_fit1, color='g', label='curve_fit')
plt.plot(x,y_fit2, color='y', label='constrained')
plt.legend(); plt.xlim(-0.1,3.5); plt.ylim(0,1.4)
print 'Exact   integral:',quad(sin ,0,pi)[0]
print 'Approx integral1:',quad(func1,0,pi,args=(popt1[0],popt1[1],
                                                popt1[2],popt1[3]))[0]
print 'Approx integral2:',quad(func1,0,pi,args=(popt2[0],popt2[1],
                                                popt2[2],popt2[3]))[0]
plt.show()

#Exact   integral: 2.0
#Approx integral1: 2.60068579748
#Approx integral2: 2.00001911981
