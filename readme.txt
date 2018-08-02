=================================== Paper reference ===================================

The companion paper for this code is:
[1]: "Projective splitting with forward steps: Asynchronous and Block-iterative Operator Splitting"
by Patrick R. Johnstone and Jonathan Eckstein.
Available at:
https://arxiv.org/abs/1803.07043

Please read this paper and familiarize yourself with it before attempting to use this code. Especially section 5 on the
numerical experiments. This code will basically allow you to replicate that experiment with your chosen data for the
lasso problem.

Please cite [1] if you use this code at all.

=================================== Install ===================================
Just copy all the python files into the same folder and run python from this folder.
This code assumes you have cvxpy installed. If you don't, update the code to not run the cvx solver by inserting the
line:

anExp.updateWhich2run({'doCVX':False})

after you have defined anExp as an instance of the class LassoExperiment.LassoFullExp (see below).


=================================== testRun.py ===================================
testRun.py features several example experiments using random data to demonstrate how the code works.


=================================== How it works in words ===================================
The key module is LassoExperiment.py which defines the class: LassoFullExp. This class defines an instance of a lasso
experiment as described in Section 5 of [1].

The lasso problem is:
min_x 0.5*||Ax - b||^2 + lambda*||x||_1

Define a blank instance using:

anExp = LassoExperiment.LassoFullExp()

----------------------------------------------------------------------------------------------------------------------
Set A and b using:

anExp.getMatvec(A,b)

----------------------------------------------------------------------------------------------------------------------
The module is based on dictionaries. All the parameters and options are defined and modified via python dictionaries.

Set lambda using

anExp.updateOptions({'lambda':5.0})

----------------------------------------------------------------------------------------------------------------------
By default, the code runs PSFor(10,G) (see Section 5 of [1]) and the cvxpy solver with lambda set to 1.0. To run, use

anExp.runTheExp()

This will plot the results as in Section 5 of [1].

----------------------------------------------------------------------------------------------------------------------
The code includes implementations of PSFor, PSBack, Fista, proximal gradient, and ADMM with the relative error
criterion of: "Relative-error approximate versions of Douglas-Rachford splitting
and special cases of the ADMM" by Eckstein and Yao. It is also includes the cvxpy solver.
To choose which methods to run use:

anExp.updateWhich2run({'doCVX': True, 'doBack':True,'doFISTA':True,'doPG':True,'doADMM':True})

Obviously set to False if you don't want to run that particular algorithm.

----------------------------------------------------------------------------------------------------------------------

To change from greedy block selection, to random, run:

anExp.updateOptions({opSelectMethod':'Random'})

To see which other options can be modified, see the self.options dictionary defined in the __init__() routine
of the class LassoFullExp in LassoExperiment.py. (line 86). The values given here are just the default values which
can be overwritten.

----------------------------------------------------------------------------------------------------------------------

To update the parameters of a given method, use the "updateParams" member function. For example, here we set
the stepsizes for PSBack:

anExp.updateParams('PSB', {'rho_0':1e-1,'rho1plus':1e-1})

----------------------------------------------------------------------------------------------------------------------
In total there are 6 python files

testRun.py:
includes several test experiments. Run this before running anything else. 

LassoExperiment.py: 
includes the main class LassoFullExp that defines an instance of the experiment. It also includes a few subroutines

runPS.py:
includes the class PSLasso with allows you to run projective splitting on the lasso problem via PSLasso.runPS()

L1LS.py:
includes implementations of FISTA and proximal gradient and a few helpful functions for the L1-regularized least squares 
(lasso) problem

proxes.py:
simply defines the prox operator for the L1 norm

algoFuncs.py:
includes several functions necessary for projective splitting
































