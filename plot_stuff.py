# data = np.loadtxt("ArgonIonization_Xsec_vs_energy.txt")
# energy = data[:,0]
# Xsec = data[:,1]
figmu = plt.figure()
axmu = figmu.add_subplot(111)
axmu.plot(EArray,mu)
axmu.set_xscale('log')
figdiff = plt.figure()
axdiff = figdiff.add_subplot(111)
axdiff.plot(EArray,Diff)
axdiff.set_xscale('log')
figalpha = plt.figure()
axalpha = figalpha.add_subplot(111)
axalpha.plot(EArray,alpha)
axalpha.set_xscale('log')
# x = np.logspace(4.0,8.0,num=100)
# y = .35*np.exp(-1.65e7/x)
# ax.plot(x,y,label='Morrow')
# ax.plot(energy,Xsec)
# ax.legend(loc=0)
plt.show()
