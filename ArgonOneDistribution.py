from bolos import parser, grid, solver
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co

# compute parameters needed for solver given common system inputs
p = 1.01e5
T = 300
N = p/(co.k*T)
xAr = 1.0

# Set-up the electron energy grid we want to compute the electron
# energy distribution on
en_start = 0.0
en_fin = 60.0
en_bins = 200
gr = grid.QuadraticGrid(en_start,en_fin,en_bins)
boltzmann = solver.BoltzmannSolver(gr)
with open('/home/lindsayad/pythonForBolos/LXCat-NoExcitation.txt') as fp:
# with open('/home/lindsayad/pythonForBolos/LXCat-June2013.txt') as fp:
    processes = parser.parse(fp)
boltzmann.load_collisions(processes)
boltzmann.target['Ar'].density = xAr
boltzmann.kT = T * co.k / solver.ELECTRONVOLT
EField = 2e5
En = EField/N
boltzmann.EN = En
boltzmann.init()
fMaxwell = boltzmann.maxwell(2.0)
f = boltzmann.converge(fMaxwell,maxn=100,rtol=1e-5)
mean_energy = boltzmann.mean_energy(f)
mu = boltzmann.mobility(f)/N
Diff = boltzmann.diffusion(f)/N
kArIz = boltzmann.rate(f,"Ar -> Ar^+")
for target,proc in boltzmann.iter_elastic():
    kArEl = boltzmann.rate(f,proc)
alpha = 1.0/(mu*En)*(xAr*kArIz)

print "mu is ",mu
print "Diff is ",Diff
print "Mean energy is ",mean_energy
print "Electron temperature is ",2.0/3*mean_energy
print "Alpha is ",alpha
print "kArEl is ",kArEl
# for target, proc in boltzmann.iter_elastic():
#     print "The rate of %s is %g" % (str(proc), boltzmann.rate(f,proc))
