# data = np.loadtxt("ArgonIonization_Xsec_vs_energy.txt")
# energy = data[:,0]
# Xsec = data[:,1]
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(EArray,alpha,label='bolos')
# ax.set_xscale('log')
x = np.logspace(4.0,8.0,num=100)
y = .35*np.exp(-1.65e7/x)
ax.plot(x,y,label='Morrow')
# ax.plot(energy,Xsec)
ax.legend(loc=0)
plt.show()
