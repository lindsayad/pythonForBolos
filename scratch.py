i = 0
d={}
for target,proc in boltzmann.iter_elastic():
    d[i]=boltzmann.rate(f,proc)
    i += 1
