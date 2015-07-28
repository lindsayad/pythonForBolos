import matplotlib.pyplot as plt
import numpy as np

# plt.clf()
etaDat = np.loadtxt('eta_data.txt')
a,b = etaDat.shape
deltaE = np.zeros(a-1)
for i in range(0,a-1):
    deltaE[i] = etaDat[i+1,0]-etaDat[i,0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
# plt.plot(deltaE,'o')
plt.plot(etaDat[:,0],'o')
plt.show()
# ax.plot(etaDat[:,0],etaDat[:,1],'o')
# ax.set_xscale('log')

# plt.show()

# import matplotlib.pyplot  as pyplot
# a = [ pow(10,i) for i in range(10) ]
# fig = pyplot.figure()
# ax = fig.add_subplot(2,1,1)

# line, = ax.plot(a, color='blue', lw=2)

# ax.set_yscale('log')

# pyplot.show()
