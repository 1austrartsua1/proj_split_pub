import LassoExperiment as LE
import numpy as np

#create A,b with random gaussian entries
n=200
d=400
A = np.random.normal(0,1,[n,d])
b = np.random.normal(0,1,[n,])

# create an instance of the LassoFullExp object
anExp = LE.LassoFullExp()

# get the random matrices A and b which define the lasso problem:
# "minimize" 0.5*np.linalg.norm(A.dot(x)-b)**2 + lam*np.linalg.norm(x,1)
anExp.getMatvec(A,b)



########################### EXPERIMENT 1 ###############################################################################
# run the experiment with DEFAULT settings
# this runs the CVXPY solver and PSFOR(10,G) with the partitions chosen at random so they all have size n/10
# PSFOr(10,G) is run with the standard backtracking procedure for selecting the forward stepsizes
# lambda = 1.0, columns of A are normalized to unit length
# you get a plot of the relative error in the function values and the subgradients both in logarithmic y-axis
#anExp.runTheExp()


########################### EXPERIMENT 2 ###############################################################################
# set lambda and also run PSBack(10,G), FISTA, proximal gradient (PG), and ADMM with relative error criterion (Eckstein Yao 2017)
# note that the simpler methods FISTA, ADMM, and PG have faster runtimes than PSFOR PSBACK because this is a very small
# example and there is significant overheads for projective splitting in creating the datastructures associated with
# each partition slice. The runtimes of the methods are comparable on larger examples of lasso.
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateWhich2run({'doBack':True,'doFISTA':True,'doPG':True,'doADMM':True})
anExp.updateOptions({'lambda':5.0})
#anExp.runTheExp()

########################### EXPERIMENT 3 ###############################################################################
# change the parameters for PSFOR. Now run PSFOR(5,10), meaning 5 partitions with randomly chosen
# block with a max delay of 10. Compare with FISTA as a baseline
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateOptions({'n_partitions':5,'opSelectMethod':'Random'})
anExp.updateParams('PSF', {'D':100})
anExp.updateWhich2run({'doFISTA':True})
#anExp.runTheExp()

########################### EXPERIMENT 4 ###############################################################################
# Run PSFOR(10,G) with the auto no-backtracking procedure for affine operators
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateParams('PSF', {'doAutoBT':True})
anExp.updateWhich2run({'doFISTA':True})
anExp.runTheExp()