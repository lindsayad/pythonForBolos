import numpy as np
import matplotlib.pyplot as plt

x = np.zeros(30)
y = np.zeros(30)
for i in range(0,30):
    x[i] = i
    y[i] = np.exp(-1/(1e-16+x[i]))
plt.plot(x,y)
plt.show()
