from bolos import parser, grid, solver
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
from collections import OrderedDict

# compute parameters needed for solver given common system inputs
p = 1.01e5
T = 300
N = p/(co.k*T)
xN2 = 0.8
xO2 = 0.2
# Set-up the electric field values we want to compute for
EStart = 2.414323855084169554e+04
EFin = 3.380053397117837518e+07
nEf = 100 # number of Electric field nodes
mult = (EFin/EStart)**(1.0/(nEf - 1.0))
EArray = np.zeros(nEf)
mu = np.zeros(nEf)
Diff = np.zeros(nEf)
alpha = np.zeros(nEf)
eta = np.zeros(nEf)
k2BodyAttach = np.zeros(nEf)
k3BodyAttach = np.zeros(nEf)
mean_energy = np.zeros(nEf)
elastic = np.zeros(nEf)

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
with open('/home/lindsayad/bolos/LXCat-June2013.txt') as fp:
    processes = parser.parse(fp)
boltzmann.load_collisions(processes)
boltzmann.target['N2'].density = xN2
boltzmann.target['O2'].density = xO2
boltzmann.kT = T * co.k / solver.ELECTRONVOLT
for i in range(0,nEf):
    EField = EArray[i]
    En = EField/N
    boltzmann.EN = En
    boltzmann.init()
    fMaxwell = boltzmann.maxwell(2.0)
    f1 = boltzmann.converge(fMaxwell,maxn=100,rtol=1e-5)
    mean_energy[i] = boltzmann.mean_energy(f1)
    mu[i] = boltzmann.mobility(f1)/N
    Diff[i] = boltzmann.diffusion(f1)/N
    kN2 = boltzmann.rate(f1,"N2 -> N2^+")
    kO2 = boltzmann.rate(f1,"O2 -> O2^+")
    k2BodyAttach[i] = boltzmann.rate(f1,"O2 -> O^-+O")
    k3BodyAttach[i] = boltzmann.rate(f1,"O2 -> O2^-")*N/(1e6)
    alpha[i] = 1.0/(mu[i]*En)*(xN2*kN2 + xO2*kO2)
    eta[i] = 1.0/(mu[i]*En)*xO2*(k2BodyAttach[i]+k3BodyAttach[i])
    j = 0
    d={}
    for target,proc in boltzmann.iter_elastic():
        d[j]=boltzmann.rate(f1,proc)
        j += 1
    kN2elastic = d[0]
    kO2elastic = d[1]
    elastic[i] = 1.0/(mu[i]*En)*(xN2*kN2elastic+xO2*kO2elastic)
D_Path = '/home/lindsayad/gdrive/programming/usefulScripts/python/'
t_vars = ['mob','diff','alpha','eta','elastic']
file_list = [D_Path + t_var + '_en.txt' for t_var in t_vars]
file_list = OrderedDict.fromkeys(file_list)
t_vars = OrderedDict.fromkeys(t_vars)
t_vars['mob'] = mu
t_vars['diff'] = Diff
t_vars['alpha'] = alpha
t_vars['eta'] = eta
t_vars['elastic'] = elastic
for f,t_var in zip(file_list.keys(),t_vars.keys()):
    with open(f,'w') as write_file:
        for i in range(0,nEf):
            write_file.write('{0:.18e}, {1:.18e}\n'.format(mean_energy[i],t_vars[t_var][i]))
    write_file.closed
# with open('diff.txt','w') as file:
#     for i in range(0,nEf):
#         file.write('{0:.18e} {1:.18e}\n'.format(mean_energy[i],Diff[i]))
# file.closed
# with open('alpha.txt','w') as file:
#     for i in range(0,nEf):
#         file.write('{0:.18e} {1:.18e}\n'.format(mean_energy[i],alpha[i]))
# file.closed
# with open('eta.txt','w') as file:
#     for i in range(0,nEf):
#         file.write('{0:.18e} {1:.18e}\n'.format(mean_energy[i],eta[i]))
# file.closed
# # with open('mean_en.txt','w') as file:
# #     for i in range(0,nEf):
# #         file.write('{0:.18e} {1:.18e}\n'.format(mean_energy[i],mean_energy[i]))
# # file.closed
