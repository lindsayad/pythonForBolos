from bolos import parser, grid, solver
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co

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
    f = boltzmann.converge(fMaxwell,maxn=100,rtol=1e-5)
    mean_energy[i] = boltzmann.mean_energy(f)
    mu[i] = boltzmann.mobility(f)/N
    Diff[i] = boltzmann.diffusion(f)/N
    kN2 = boltzmann.rate(f,"N2 -> N2^+")
    kO2 = boltzmann.rate(f,"O2 -> O2^+")
    k2BodyAttach[i] = boltzmann.rate(f,"O2 -> O^-+O")
    k3BodyAttach[i] = boltzmann.rate(f,"O2 -> O2^-")*N/(1e6)
    alpha[i] = 1.0/(mu[i]*En)*(xN2*kN2 + xO2*kO2)
    eta[i] = 1.0/(mu[i]*En)*xO2*(k2BodyAttach[i]+k3BodyAttach[i])
with open('td_air.txt','w') as file:
    file.write('efield[V/m]_vs_mu[m2/Vs]\n')
    file.write('AIR\nCOMMENT: data generated by bolos\n')
    file.write('-----------------------\n')
    for i in range(0,nEf):
        file.write('{0:.18e} {1:.18e}\n'.format(EArray[i],mu[i]))
    file.write('-----------------------\n\n\n\n')
    file.write('efield[V/m]_vs_dif[m2/s]\n')
    file.write('AIR\nCOMMENT: data generated by bolos\n')
    file.write('-----------------------\n')
    for i in range(0,nEf):
        file.write('{0:.18e} {1:.18e}\n'.format(EArray[i],Diff[i]))
    file.write('-----------------------\n\n\n\n')
    file.write('efield[V/m]_vs_alpha[1/m]\n')
    file.write('AIR\nCOMMENT: data generated by bolos\n')
    file.write('-----------------------\n')
    for i in range(0,nEf):
        file.write('{0:.18e} {1:.18e}\n'.format(EArray[i],alpha[i]))
    file.write('-----------------------\n\n\n\n')
    file.write('efield[V/m]_vs_eta[1/m]\n')
    file.write('AIR\nCOMMENT: data generated by bolos\n')
    file.write('-----------------------\n')
    for i in range(0,nEf):
        file.write('{0:.18e} {1:.18e}\n'.format(EArray[i],eta[i]))
    file.write('-----------------------\n\n\n\n')
    file.write('efield[V/m]_vs_energy[eV]\n')
    file.write('AIR\nCOMMENT: data generated by bolos\n')
    file.write('-----------------------\n')
    for i in range(0,nEf):
        file.write('{0:.18e} {1:.18e}\n'.format(EArray[i],mean_energy[i]))
    file.write('-----------------------\n\n\n\n')
file.closed


# newgrid = grid.QuadraticGrid(en_start,en_fin,en_bins)
# boltzmann.grid = newgrid
# boltzmann.init()
# finterp = boltzmann.grid.interpolate(f,gr)
# f1 = boltzmann.converge(finterp,maxn=200,rtol=1e-5)
# # plt.plot(boltzmann.cenergy,f1,label='f1')
# mean_energy_f1 = boltzmann.mean_energy(f1)
# boltzmann.target['N2'].density = 1.0
# boltzmann.target['O2'].density = 0.0
# boltzmann.init()
# f2 = boltzmann.converge(finterp,maxn=200,rtol=1e-5)
# plt.plot(boltzmann.cenergy,f2,label='Nitrogen only')
# mean_energy = boltzmann.mean_energy(f2)
# plt.title('EEDF')
# plt.legend(loc=1)
# plt.xlabel('Energy (ev)')
# plt.ylabel('f(E)')
# plt.show()

