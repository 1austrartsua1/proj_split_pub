import LassoExperiment as LE
import numpy as np

#create A,b,lambda
n=100
d=200
A = np.random.normal(0,1,[n,d])
b = np.random.normal(0,1,[n,])
lam = 1e0

anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.runTheExp()