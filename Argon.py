# from bolos import parser, grid, solver
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.constants as co
# from collections import OrderedDict
# import os

# # compute parameters needed for solver given common system inputs
# p = 1.01e5
# T = 300
# N = p/(co.k*T)
# xAr = 1.0
# # Set-up the electric field values we want to compute for
# EStart = 1.0e+04
# EFin = 1.0e+07
# nEf = 200 # number of Electric field nodes
# mult = (EFin/EStart)**(1.0/(nEf - 1.0))
# EArray = np.zeros(nEf)
# tdArray = np.zeros(nEf)
# mu = np.zeros(nEf)
# Diff = np.zeros(nEf)
# alpha = np.zeros(nEf)
# eta = np.zeros(nEf)
# mean_energy = np.zeros(nEf)
# kArIz = np.zeros(nEf)
# kArEx = np.zeros(nEf)
# kArEl = np.zeros(nEf)

# EArray[0] = EStart
# tdArray[0] = EStart/(N*1.0e-21)
# for i in range(0,nEf-1):
#     EArray[i+1]=EArray[i]*mult
#     tdArray[i+1]=EArray[i+1]/(N*1.0e-21)
    

# # Set-up the electron energy grid we want to compute the electron
# # energy distribution on
# en_start = 0.0
# en_fin = 60.0
# en_bins = 200
# gr = grid.QuadraticGrid(en_start,en_fin,en_bins)
# boltzmann = solver.BoltzmannSolver(gr)
# # with open('/mnt/Data/AlexApps/pythonForBolos/LXCat-June2013.txt') as fp:
# with open(os.path.expandvars('${BOLOSDIR}/LXCat-June2013.txt')) as fp:
#     processes = parser.parse(fp)
# boltzmann.load_collisions(processes)
# boltzmann.target['Ar'].density = xAr
# boltzmann.kT = T * co.k / solver.ELECTRONVOLT
# for i in range(0,nEf):
#     EField = EArray[i]
#     En = EField/N
#     boltzmann.EN = En
#     boltzmann.init()
#     fMaxwell = boltzmann.maxwell(5.0)
#     if i==0:
#         f = boltzmann.converge(fMaxwell,maxn=100,rtol=1e-5)
#     else:
#         f = boltzmann.converge(f,maxn=100,rtol=1e-6)
#     mean_energy[i] = boltzmann.mean_energy(f)
#     mu[i] = boltzmann.mobility(f)/N
#     Diff[i] = boltzmann.diffusion(f)/N
#     kArIz[i] = boltzmann.rate(f,"Ar -> Ar^+")
#     kArEx[i] = boltzmann.rate(f,"Ar -> Ar*(11.5eV)")
#     alpha[i] = 1.0/(mu[i]*En)*(xAr*kArIz[i])
#     for target,proc in boltzmann.iter_elastic():
#         kArEl[i] = boltzmann.rate(f,proc)

# # plt.plot(tdArray,kArIz,label='Ionization')
# # plt.plot(tdArray,kArEx,label='Excitation')
# # plt.plot(tdArray,kArEl,label='Elastic')
# # plt.plot(mean_energy,kArIz,label='Ionization')
# # plt.plot(mean_energy,kArEx,label='Excitation')
# # plt.plot(mean_energy,kArEl,label='Elastic')
# # plt.yscale('log')
# # plt.xscale('log')
# # plt.legend(loc=0)
# # plt.xlim(4,np.max(tdArray))
# # plt.ylim(1e-30,1e-12)
# # plt.show()

# for i in range(len(alpha)):
#     if alpha[i] < 1e-15:
#         alpha[i] = 0.

# mean_energy = np.insert(mean_energy,0,0.)
# alpha = np.insert(alpha,0,0.)

# x = mean_energy
# y = alpha
# dx1 = np.zeros(x.shape)
# dx2 = np.zeros(x.shape)
# a = np.zeros(x.shape)
# b = np.zeros(x.shape)
# c = np.zeros(x.shape)
# yprime = np.zeros(x.shape)

# for i in range(1,x.size-1):
#     dx1[i] = x[i]-x[i-1]
#     dx2[i] = x[i+1]-x[i]
#     a[i] = -dx2[i]/(dx1[i]*(dx1[i]+dx2[i]))
#     b[i] = (dx2[i]-dx1[i])/(dx1[i]*dx2[i])
#     c[i] = dx1[i]/(dx2[i]*(dx1[i]+dx2[i]))
#     yprime[i] = a[i]*y[i-1]+b[i]*y[i]+c[i]*y[i+1]                   

# dx2[0] = x[1]-x[0]
# a[0] = -(2*dx2[0]+dx2[1])/(dx2[0]*(dx2[0]+dx2[1]))
# b[0] = (dx2[0]+dx2[1])/(dx2[0]*dx2[1])
# c[0] = -dx2[0]/(dx2[1]*(dx2[0]+dx2[1]))
# yprime[0] = a[0]*y[0]+b[0]*y[1]+c[0]*y[2]

# dx1[x.size-1] = x[x.size-1]-x[x.size-2]
# a[x.size-1] = dx1[x.size-1]/(dx1[x.size-2]*(dx1[x.size-2]+dx1[x.size-1]))
# b[x.size-1] = -(dx1[x.size-1]+dx1[x.size-2])/(dx1[x.size-1]*dx1[x.size-2])
# c[x.size-1] = (2*dx1[x.size-1]+dx1[x.size-2])/(dx1[x.size-1]*(dx1[x.size-1]+dx1[x.size-2]))
# yprime[x.size-1] = a[x.size-1]*y[x.size-3] + b[x.size-1]*y[x.size-2] + c[x.size-1]*y[x.size-1]

D_Path = os.path.expandvars('${ZAPDIR}/src/materials/')
f = D_Path + "td_argon_mean_en.txt"
with open(f,'w') as write_file:
    for i in range(0,nEf):
        write_file.write('{0:.18e} {1:.18e} {2:.18e}\n'.format(mean_energy[i], alpha[i], yprime[i]))

# t_vars = ['kAr','alpha']
# file_list = [D_Path + t_var + '_LFA.txt' for t_var in t_vars]
# file_list = OrderedDict.fromkeys(file_list)
# t_vars = OrderedDict.fromkeys(t_vars)
# # t_vars['mob'] = mu
# # t_vars['diff'] = Diff
# t_vars['alpha'] = alpha
# t_vars['kAr'] = kAr
# # t_vars['eta'] = eta
# # t_vars['elastic'] = elastic
# for f,t_var in zip(file_list.keys(),t_vars.keys()):
#     with open(f,'w') as write_file:
#         for i in range(0,nEf):
#             write_file.write('{0:.18e}, {1:.18e}\n'.format(EArray[i],t_vars[t_var][i]))
#     write_file.closed
