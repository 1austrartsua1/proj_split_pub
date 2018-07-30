# Copyright 2018, Patrick R. Johnstone
#    This toolbox is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License <http://www.gnu.org/licenses/> for more
#    details.

## paper reference
# Projective Splitting with Forward Steps: Asynchronous and Block Iterative Operator Splitting
# Patrick R. Johnstone and Jonathan Eckstein, arXiv:1803.07043, March 2018
#

# Throughout we use dictionaries for the options of each method and the overall experiment
# these are a very convenient python data structure. You can update the dictionary using .update(newDict)
# if unfamiliar with python dictionaries, one should read about them online before attempting to use this code

# see testRun.py for examples and an explanation of how to use the code

import numpy as np

import runPS as ps
import L1LS as LSmod
import cvxpy as cvx
#If you don't have cvxpy and don't want to download, comment this out. Also set 'doCVX' field in the
#which2run dictionary to False

import proxes as px

import time
from matplotlib import pyplot as plt

class OutObj:
    # this is an object to contain the outputs of an algorithm
    #1. the function values at each iteration
    #2. the number of  Q-equivalent matrix multiplies per iteration
    #3. clock runtime
    #4. the subgradient values per iteration

    def __init__(self,f,mults,runtime,subgs):
        self.fout = f
        self.mults = mults
        self.runtime = runtime
        self.sgout = subgs

class MethodParams:
    #parameters for each algorithm
    #stored as a dictionary
    def __init__(self,algo):
        if(algo=='PSF'):
            self.params ={'doForward':True,'D':0}
        elif(algo=='PSB'):
            self.params = {'D':0}
        elif(algo=='Fista'):
            self.params = {'doBT':True,'maxIter':200,'intialStep':1e0}
            #doBT means to do backtracking
        elif(algo=='PG'):
            self.params = {'maxIter': 200}
        elif(algo=='ADMM'):
            self.params = {'maxIter': 100,'c':1.0,'sigma':0.9,'maxInner':20,'doP':False,'doTheta':True}


    def updateParams(self,newParams):
        self.params.update(newParams)
        #update the parameters of a given method.



class LassoFullExp:
    #this class defines an instance of the lasso experiment from the paper
    #in addition to running projective splitting we also implement proximal gradient, FISTA, and ADMM with
    # the relative error criterion (Eckstein: 2016).
    # We also use the CVXPY solver. If you don't have CVXPY, comment out
    def __init__(self):
        #default values for options
        # which algorithms to run
        # default is PSFor and CVX
        self.which2run =\
        {
            'doCVX': True, # set to false if you don't have the CVXPY module
            'doFor': True, # Projective Splitting with forward steps
            'doBack': False, # Projective Splitting with backward steps
            'doFISTA': False,
            'doPG': False,
            'doADMM': False
        }
        # these are the default values for the options. Update them using "updateOptions" member function.
        self.options =\
        {
        'lam':1.0,
        'opSelectMethod': 'Greedy', # greedy block selection. Can also set to: 'Random'.
        'partition_type': 'random', # can also set to 'Uniform'.
        'n_partitions':10,
        'normalize':True, #normalize the columns of the data matrix to have unit norm
        'funcVals':True, #plot function values
        'rawIter':False, #If true use actual iterations for the x-axis of the plot. Otherwise use Q-equivalent multiplies
        'rawFuncPlt':False, # If False, plot the function values on a logarithmic scale
        'subgVals':True, # If true, plot the subgradient values
        'plotSetLims':False, # If true, set limits on the x-axis and y-axis for the plots
        'verbose':True # If true print out iteration counter
        }
        self.pSFparams = MethodParams('PSF')
        self.pSBparams = MethodParams('PSB')
        self.fistaparams = MethodParams('Fista')
        self.pGparams = MethodParams('PG')
        self.aDMMparams = MethodParams('ADMM')

    def updateParams(self, method, newParams):
        #wrapper function to modify method's parameters
        if(method=='PSF'):
            self.pSFparams.updateParams(newParams)
        elif(method=='PSB'):
            self.pSBparams.updateParams(newParams)
        elif(method=='Fista'):
            self.fistaparams.updateParams(newParams)
        elif(method=='PG'):
            self.pGparams.updateParams(newParams)
        elif (method == 'ADMM'):
            self.aDMMparams.updateParams(newParams)


    def getData(self,A,b,partition_list):
        # Unlike the paper, we use A for the data matrix rather than Q
        self.A = A
        self.b = b
        self.partition_list = partition_list

    def updateOptions(self,newOptions):
        self.options.update(newOptions)


    def updateWhich2run(self,newRuns):
        self.which2run.update(newRuns)

    def runWoptions(self,method):
        # run a specific method with the options given in the options dictionarys
        if((method=='PSF')|(method=='PSB')):
            newOptions = self.options.copy()
            # this object prepares to run projective splitting
            outObj = ps.PSLasso()
            outObj.getData(self.A, self.b, self.options['lam'], self.partition_list)
            if(method=='PSF'):
                newOptions.update(self.pSFparams.params)
            else:
                newOptions.update(self.pSBparams.params)
            outObj.updateOptions(newOptions)
            print"running "+method

            # run projective splitting
            outObj.runPS()


            print method+' running time ' + str(outObj.runtime)
            print 'sparsity '+str(outObj.sparsity)
        elif(method=='fista'):
            outObj = self.runFista()
        elif(method =='PG'):
            print('running Prox Grad')
            tpg = time.time()
            [_,n_cols]=self.A.shape
            [_, f_pg, mults_pg,pg_subg,t_subg] = LSmod.proxGradL1LS(self.A,self.b, self.options['lam'], np.zeros(n_cols), 1.0,
                                                     self.pGparams.params['maxIter'], 'backTrack',
                                                     self.options['subgVals'],self.options['verbose'])

            tpg = time.time() - tpg
            outObj = OutObj(f_pg, mults_pg, tpg-t_subg,pg_subg)
            print 'prox grad running time ' + str(tpg-t_subg)
        elif(method == 'ADMM'):
            print"running ADMM"
            tadmm = time.time()
            [ftheta, fp, theta, p, ADMM_mults,theta_subg,p_subg,tadmmTermC] = \
            ADMMRelErr_eNet(self.A, self.b, self.options['lam'],
                               self.aDMMparams.params['maxIter'], self.aDMMparams.params['c'],
                               self.aDMMparams.params['maxInner'],self.aDMMparams.params['sigma'],
                               self.options['subgVals'],self.aDMMparams.params['doP'],
                               self.aDMMparams.params['doTheta'],self.options['funcVals'],self.options['verbose'])
            tadmm = time.time() - tadmm
            print 'ADMM running time ' + str(tadmm-tadmmTermC)
            print"ADMM sparsity "+str(sum(abs(theta)>1e-5))

            outObj = OutObj(ftheta, ADMM_mults, tadmm-tadmmTermC,theta_subg)
            self.outADMMrelp = OutObj(fp, ADMM_mults, tadmm,p_subg)

        return outObj

    def runTheExp(self):
        # This function actually runs the full lasso experiment with all the options and plots the results

        #first create the partition to be used by projective splitting
        [partition_listNew, _] = createPartitions(self.options['n_rows'], self.options['n_partitions'],
                                                     self.options['partition_type'])

        if(self.options['opSelectMethod']=='Greedy'):
            # determine the name for projective splitting in the legend of the plot
            self.updateParams('PSF', {'name':'PS'+'F'+'('+str(self.options['n_partitions'])+',G)'})
            self.updateParams('PSB', {
                'name': 'PS' + 'B' + '(' + str(self.options['n_partitions']) + ',G)'})

        else:
            self.updateParams('PSF', {'name': 'PS' + 'F' + '(' + str(self.options['n_partitions']) + ',' + str(
                self.pSFparams.params['D']) + ')'})
            self.updateParams('PSB', {
                'name': 'PS' + 'B' + '(' + str(self.options['n_partitions']) + ',' + str(
                    self.pSBparams.params['D']) + ')'})

        self.getData(self.A0,self.b0,partition_listNew)
        if(self.options['normalize']):
            self.normalizeCols()

        self.runSingleExp()
        print"========================="
        self.metaPlot()



    def runSingleExp(self):
        runTime = time.time()
        [_,n_cols] = self.A.shape

        if(self.which2run['doCVX']):
            print"========================="
            print('Running CVX opt...')
            tcvx = time.time()
            [self.cvxopt,_] = cvxSolL1LS(n_cols,self.b,self.A,self.options['lam'])
            tcvx = time.time() - tcvx
            print'cvx running time '+str(tcvx)


        if(self.which2run['doFor']):
            print"========================="
            self.PSFor = self.runWoptions('PSF')

        if(self.which2run['doBack']):
            print"========================="
            self.PSBack = self.runWoptions('PSB')

        if(self.which2run['doFISTA']):
            print"========================="
            self.outFista = self.runWoptions('fista')

        if(self.which2run['doPG']):
            print"========================="
            self.outPG = self.runWoptions('PG')

        if(self.which2run['doADMM'] ==True):
            print"========================="
            self.outADMMrelThta = self.runWoptions('ADMM')


        if(self.options['funcVals']):
            self.opt = self.getopt()

        runTimeT = time.time() - runTime
        print"Total run time: " + str(runTimeT)
        return


    def getopt(self):
        # get the optimal function value which is estimated as the smallest value encountered by any of the methods
        listoMins = []
        if(self.which2run['doCVX']):
            listoMins.append(self.cvxopt)
        if(self.which2run['doFor']):
            listoMins.append(min(self.PSFor.fout))
        if(self.which2run['doBack']):
            listoMins.append(min(self.PSBack.fout))
        if (self.which2run['doFISTA']):
            listoMins.append(min(self.outFista.fout))
        if(self.which2run['doPG']):
            listoMins.append(min(self.outPG.fout))
        if (self.which2run['doADMM'] == True):
            listoMins.append(min(self.outADMMrelThta.fout))
            listoMins.append(min(self.outADMMrelp.fout))
        return min(listoMins)


    def runFista(self):
        [n_rows,n_cols] = self.A.shape

        if (self.fistaparams.params['doBT'] == False):
            svdobject = np.linalg.svd(self.A)
            L = svdobject[1][0] ** 2
            tau = 1 / L
        else:
            tau = self.fistaparams.params['intialStep']
        x0 = np.zeros(n_cols)
        print('running FISTA')
        tfista = time.time()
        [f_fista, x, iterFista,subgsFista,tsubgs] = LSmod.FISTAL1LS(self.A, self.b, self.options['lam'], x0, tau,
                                                  self.fistaparams.params['maxIter'],False, -1,
                                                  self.fistaparams.params['doBT'],self.options['subgVals'],
                                                    self.options['verbose'])
        tfista = time.time() - tfista - tsubgs
        print 'fista running time ' + str(tfista)
        sparsityFista = sum(abs(x)>1e-5)
        print 'fista sparsity '+str(sparsityFista)
        return OutObj(f_fista,iterFista,tfista,subgsFista)


    def getMatvec(self,Ain,b_in):
        #this is how you read the data matrix and vector in
        self.A0 = Ain
        self.b0 = b_in
        [n_rows,_] = Ain.shape
        self.updateOptions({'n_rows':n_rows})


    def normalizeCols(self):
        [nrows,ncols] = self.A.shape
        for i in range(ncols):
            if(np.linalg.norm(self.A.T[i])>1e-5):
                self.A.T[i] = self.A.T[i]/(np.linalg.norm(self.A.T[i]))


    def doAplot(self,obj,name):
        #do a plot for a single algorithm

        if(self.options['rawIter']):
            plotx = obj.iters
        else:
            plotx = obj.mults


        self.legendList.append(name)
        self.maxMult.append(max(plotx))
        if (self.options['funcVals']):

            ploty = obj.fout
            if(self.options['rawFuncPlt']):
                self.FigFunc.plot(plotx,ploty)
            else:
                plotx = np.array(plotx)
                ploty = np.array(ploty)
                self.FigFunc.plot(plotx, (ploty - self.opt)/self.opt)



        if (self.options['subgVals']):
            ploty = obj.sgout
            self.FigSG.plot([plotx[i] for i in range(len(plotx))], [ploty[i] for i in range(len(ploty))])


    def metaPlot(self):
        # plot all algorithms
        figFunc = plt.figure()
        figSG = plt.figure()
        self.FigFunc = figFunc.add_subplot(111)
        self.FigSG = figSG.add_subplot(111)

        self.maxMult = []
        self.legendList = []
        if (self.which2run['doFor']):
            self.doAplot(self.PSFor, self.pSFparams.params['name'])

        if (self.which2run['doBack']):
            self.doAplot(self.PSBack, self.pSBparams.params['name'])

        if (self.which2run['doFISTA']):
            self.doAplot(self.outFista, 'FISTA')

        if (self.which2run['doPG']):
            self.doAplot(self.outPG, 'PG')

        if (self.which2run['doADMM']):
            if (self.aDMMparams.params['doTheta']):
                self.doAplot(self.outADMMrelThta, 'RE-ADMM')
            if (self.aDMMparams.params['doP']):
                self.doAplot(self.outADMMrelp, 'RE-ADMMp')

        self.FigFunc.legend(self.legendList)
        self.FigSG.legend(self.legendList)
        self.FigSG.set_yscale('log')

        if(self.options['plotSetLims']):
            self.FigFunc.set_xlim(self.options['xlim'])
            self.FigFunc.set_ylim(self.options['ylim'])
            self.FigSG.set_ylim(self.options['ylimSG'])
            self.FigSG.set_xlim(self.options['xlim'])

        if(self.options['rawIter']):
            self.FigFunc.set_xlabel('Iteration count')
            self.FigSG.set_xlabel('Iteration count')
        else:
            self.FigFunc.set_xlabel('Q equivalent multiplies')
            self.FigSG.set_xlabel('Q equivalent multiplies')

        self.FigFunc.set_ylabel('relative error function values')
        self.FigSG.set_ylabel('minimum norm of subgradient')

        if (self.options['rawFuncPlt'] == False):
            self.FigFunc.set_yscale('log')

        plt.show()



def fval(theta,A,b,lam):
    #Evaluate the lasso function
    return (0.5*np.linalg.norm(A.dot(theta)-b)**2+lam*np.linalg.norm(theta,1) )


def ADMMRelErr_eNet(A,b,lam1,maxIter,c,maxInner,sigma,doSubg,doP,doTheta,doFuncs,verbose):
    [_,d] = A.shape
    p = np.zeros(d)
    theta = np.zeros(d)
    nu = np.zeros(d)
    Atb = A.T.dot(b)
    fp = [fval(p, A, b, lam1)]
    ftheta = [fval(theta, A, b, lam1)]
    p_subg = [LSmod.maxSubg(A,b,lam1,p)]
    theta_subg = [LSmod.maxSubg(A,b,lam1,theta)]
    ts_termC = 0
    MatMults = [0]
    for k in range(maxIter):
        if(verbose):
            print"ADMM Iteration: "+str(k)
        nmults = 0
        bright = Atb + c*theta - nu
        Ap = A.dot(p)
        Aleftp = A.T.dot(Ap) + c*p
        r = bright -Aleftp
        nmults += 2
        q = r
        pl = p
        for l in range(maxInner):
            [pl,r,q] = pUpdateADMM(A,r,q,pl,c)
            nmults += 2
            nul = nu + c*(pl - theta) + r
            thetal = thetaUpdateADMM(lam1,c,pl,nul)
            if(np.linalg.norm(r)<=sigma*np.linalg.norm(nul - nu - c*(thetal - theta))):
                break
        MatMults.append(MatMults[len(MatMults)-1]+nmults)
        p = pl
        nu = nu + c*(pl - theta)
        theta = thetal
        t_termC = time.time()
        if(doSubg):
            if(doP):
                p_subg.append(LSmod.maxSubg(A,b,lam1,p))
            if(doTheta):
                theta_subg.append(LSmod.maxSubg(A,b,lam1,theta))
        if(doFuncs):
            if(doP):
                fp.append(fval(p, A, b, lam1))
            if(doTheta):
                ftheta.append(fval(theta, A, b, lam1))
        t_termC = time.time() - t_termC
        ts_termC += t_termC
    return [ftheta,fp,theta,p,MatMults,theta_subg,p_subg,ts_termC]


def pUpdateADMM(A,r,q,p,c):
    Aq = A.dot(q)
    Aleftq = A.T.dot(Aq) + c * q

    alpha =r.T.dot(r)/(q.T.dot(Aleftq))
    p = p + alpha*q
    rnew = r - alpha*Aleftq
    beta = rnew.T.dot(rnew)/(r.T.dot(r))
    q = rnew + beta*q
    r = rnew
    return [p,r,q]


def thetaUpdateADMM(lam1,c,p,nu):
    return px.proxL1(p+nu/c,lam1/c,1)



def createPartitions(n_rows,n_partitions,partition_choice):

    partition_size = float(n_rows) / float(n_partitions)
    partition_size2use = int(partition_size)

    if(partition_choice=='uniform'):
        partition_list = [range(i, i + partition_size2use) for i in range(0, partition_size2use*(n_partitions-1)+1, partition_size2use)]
        if (partition_size != int(partition_size)):
            partition_list.append(range(partition_size2use*n_partitions,n_rows))
    else:
        ind = np.random.permutation(n_rows)

        partition_list = [ind[range(i, i + partition_size2use)] for i in range(0, partition_size2use*(n_partitions-1)+1, partition_size2use)]
        if (partition_size != int(partition_size)):
            partition_list.append(ind[range(partition_size2use * n_partitions, n_rows)])
    partition_list = [[partition_list[i][j] for j in range(len(partition_list[i]))] for i in range(len(partition_list))]

    return [partition_list,partition_size]


def cvxSolL1LS(n_cols,b,A,lam):
    x_cvx = cvx.Variable(n_cols)
    bmat = np.matrix(b)
    bmat = bmat.T
    Amat = np.matrix(A)
    f = cvx.sum_squares(Amat * x_cvx - bmat) / 2 + lam * cvx.norm(x_cvx, 1)    
    problem = cvx.Problem(cvx.Minimize(f))
    problem.solve()
    opt = problem.value
    xopt = x_cvx.value
    NNZ = sum([abs(xopt[i])>1e-6 for i in range(xopt.shape[0])])
    print('Number nonzero in solution is ' + str(float(NNZ)) + ' out of '+ str(n_cols) )
    return [opt,xopt]



