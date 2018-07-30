# Copyright 2018, Patrick R. Johnstone
#    This toolbox is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License <http://www.gnu.org/licenses/> for more
#    details.

import LassoExperiment as LE
import numpy as np



#create A,b with random Gaussian entries
n=200
d=400
A = np.random.normal(0,1,[n,d])
b = np.random.normal(0,1,[n,])

# create an instance of the LassoFullExp object
anExp = LE.LassoFullExp()

# get the matrix A and vector b which define the lasso problem:
# "minimize" 0.5*np.linalg.norm(A.dot(x)-b)**2 + lam*np.linalg.norm(x,1)
anExp.getMatvec(A,b) # given a data matrix A and vector b, this is how you update them in the code



########################### EXPERIMENT 1 ###############################################################################
# run the experiment with DEFAULT settings
# this runs the CVXPY solver and PSFOR(10,G) with the partitions chosen at random so they all have size n/10
# PSFOr(10,G) is run with the standard backtracking procedure for selecting the forward stepsizes
# lambda = 1.0, columns of A are normalized to unit length
# you get a plot of the relative error in the function values and the subgradients both in logarithmic y-axis
anExp.runTheExp()


########################### EXPERIMENT 2 ###############################################################################
# set lambda and also run PSBack(10,G), FISTA, proximal gradient (PG), and ADMM with relative error criterion (Eckstein Yao 2017)
# note that the simpler methods FISTA, ADMM, and PG have faster runtimes than PSFOR PSBACK because this is a very small
# example and there is significant overheads for projective splitting in creating the datastructures associated with
# each partition slice. The runtimes of the methods are comparable on larger examples of lasso.
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateWhich2run({'doBack':True,'doFISTA':True,'doPG':True,'doADMM':True})
anExp.updateOptions({'lambda':5.0,'verbose':False}) # this is how you set the value of lambda. Also, setting verbose
#to false supresses some of the print statements
anExp.runTheExp()

########################### EXPERIMENT 3 ###############################################################################
# change the parameters for PSFOR. Now run PSFOR(5,10), meaning 5 partitions with randomly chosen
# block with a max delay of 10. Compare with FISTA as a baseline
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateOptions({'n_partitions':5,'opSelectMethod':'Random'})
anExp.updateParams('PSF', {'D':100})
anExp.updateWhich2run({'doFISTA':True})
anExp.runTheExp()

########################### EXPERIMENT 4 ###############################################################################
# Run PSFOR(10,G) with the auto no-backtracking procedure for affine operators
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateParams('PSF', {'doAutoBT':True})
anExp.updateWhich2run({'doFISTA':True})
anExp.runTheExp()

########################### EXPERIMENT 5 ###############################################################################
# Run PSFor(10,G) with fixed arbitrarily chosen stepsize (no backtracking)
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateWhich2run({'doCVX':False})
anExp.updateParams('PSF', {'rho_0':1e-1,'rho1plus':1e-1,'doBackTrack':False}) # this sets the fixed values for the stepsizes
# rho_0 corresponds to the ell_1 term (note this is different to the notation in the paper). rho_1plus are
#the stepsizes corresponding to all the least-squares terms. For simplicity this implementation does
#not allow each least squares slice to use a different stepsize
anExp.runTheExp()


########################### EXPERIMENT 6 ###############################################################################
# Run PSFor(10,G) but use make sure that every M=100 iterations every block has been updated at least once
# while this does not typically help empirical performance, it is necessary to apply the theorem from the paper
# which implies iterate convergence
# note that technically it may take M+n_partitions iterations for every operator to be updated. But the important point
# is that it is bounded which is what matters for the convergence theory

anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateWhich2run({'doCVX':False})
anExp.updateParams('PSF', {'upperGreedy':True,'M':25})
anExp.runTheExp()

########################### EXPERIMENT 7 ###############################################################################
# run PSFOr(10,G) but evaluate the prox for the ell_1 only once every 10 iterations
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateWhich2run({'doCVX':False})
anExp.updateParams('PSF', {'L1every':False,'L1freq':5})
anExp.runTheExp()


########################### EXPERIMENT 8 ###############################################################################
# run PSFOr(10,G) and plot with the x-axis as raw iterations not "Q-equivalent multiplies"
anExp = LE.LassoFullExp()
anExp.getMatvec(A,b)
anExp.updateWhich2run({'doCVX':False})
anExp.updateOptions({'rawIter':True})
anExp.runTheExp()
