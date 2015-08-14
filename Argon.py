from bolos import parser, grid, solver
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
from collections import OrderedDict
import os

# compute parameters needed for solver given common system inputs
p = 133
T = 400
N = p/(co.k*T)
xAr = 1.0
# Set-up the electric field values we want to compute for
EStart = 1.0e+03
EFin = 1.0e+06
nEf = 200 # number of Electric field nodes
mult = (EFin/EStart)**(1.0/(nEf - 1.0))
EArray = np.zeros(nEf)
mu = np.zeros(nEf)
Diff = np.zeros(nEf)
alpha = np.zeros(nEf)
eta = np.zeros(nEf)
mean_energy = np.zeros(nEf)
kAr = np.zeros(nEf)

EArray[0] = EStart
for i in range(0,nEf-1):
    EArray[i+1]=EArray[i]*mult

# Set-up the electron energy grid we want to compute the electron
# energy distribution on
en_start = 0.0
en_fin = 60.0
en_bins = 200
gr = grid.QuadraticGrid(en_start,en_fin,en_bins)
boltzmann = solver.BoltzmannSolver(gr)
with open('/mnt/Data/AlexApps/pythonForBolos/LXCat-June2013.txt') as fp:
    processes = parser.parse(fp)
boltzmann.load_collisions(processes)
boltzmann.target['Ar'].density = xAr
boltzmann.kT = T * co.k / solver.ELECTRONVOLT
for i in range(0,nEf):
    EField = EArray[i]
    En = EField/N
    boltzmann.EN = En
    boltzmann.init()
    fMaxwell = boltzmann.maxwell(5.0)
    if i==0:
        f = boltzmann.converge(fMaxwell,maxn=100,rtol=1e-5)
    else:
        f = boltzmann.converge(f,maxn=100,rtol=1e-6)
    mean_energy[i] = boltzmann.mean_energy(f)
    mu[i] = boltzmann.mobility(f)/N
    Diff[i] = boltzmann.diffusion(f)/N
    kAr[i] = boltzmann.rate(f,"Ar -> Ar^+")
    alpha[i] = 1.0/(mu[i]*En)*(xAr*kAr[i])
D_Path = os.path.expandvars('${ZAPDIR}/src/materials/')
f = D_Path + "td_argon.txt"
with open(f,'w') as write_file:
    for i in range(0,nEf):
        write_file.write('{0:.18e} {1:.18e} {2:.18e} {3:.18e}\n'.format(EArray[i], mu[i], Diff[i], alpha[i]))



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
