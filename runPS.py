# Copyright 2018, Patrick R. Johnstone
#    This toolbox is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License <http://www.gnu.org/licenses/> for more
#    details.

import numpy as np
import L1LS as LSmod
import algoFuncs as af
import proxes as px
import time



class PSLasso:
    def __init__(self):
        # these are the default options that can be overwritten
        # they are written in an options dictionary
        self.options = {
        'doBackTrack':True,
        'maxIter':1000,
        'rho_0':1.0, # stepsize corresponding to the ell-1 term
		'rho1plus':1.0, #stepsizes corresponding to the least squares terms, all the same for simplicity
        'doForward':False, # forward or backward steps for the least-squares terms
        'Delta':1.0,
        'beta':1.0,
        'sigma':0.95,
        'gamma':1.0,
        'doErrCheck':True, # for the backward steps, use the relative error criteria
        'maxInner':20, # max inner iterations for CG in backward steps
        'funcIndex':0, # this is the x slice at which we choose to compute function values and subgradients
                       # set to -1 to calculate at $z^k$.
        'opSelectMethod':'Greedy', # method for selecting slice to update (Random, Full, Greedy)
        'opPerIter':1, # how many slices to update per iteration
        'randomOpAdd': False, # whether to add a random number of slices per iteration
        'D':0,       # max amount of delay
        'reducedW':True, # our hyplerplane formulation. If False use Eckstein (2017) formulation
        'doAutoBT':False, # whether to use the automatic backtracking stepsize for affine operators
		'equalizeRho':True, # when doing forward steps with backtracking, set the stepsize to be the average of the 
		                    # least squares stepsizes
        'upperGreedy':False, # If set to true, then when using greedy selection, check whether each block has been
                             # updated every M iterations
        'M':1, # see 'upperGreedy'
        'funcVals':True, #get plots of function values
        'subgVals':False, #get plots of subgradient values
        'L1every':True, #do the prox for the L1 every iteration
        'L1freq':0 #If L1every is false, then do a L1 backward step every 'L1freq' iterations. If this is set to 0
        # add the l1 term to the greedy search
        }

    def getData(self,Amtx,bvec,lambdaval,part_list_in):
        self.A = Amtx
        self.b = bvec
        self.lam = lambdaval
        self.partition_list = part_list_in


    def updateOptions(self,newOps):
        self.options.update(newOps)

    def runPS(self):
        t=time.time()

        if(self.options['doForward']==True):
            #for printing purposes
            string='F'
        else:
            string='B'

        self.createListofMats()

        m = len(self.partition_list)
        [n_rows,n_cols] = self.A.shape

        #precompute A transpose b for each slice
        self.getAtbs(m)


        #stepsizes
        rho = [self.options['rho_0']]
        rho.extend([self.options['rho1plus'] for j in range(m)])

        # initialization
        self.createDataStructs(m,n_cols)

        if(self.options['L1every']):
            n2use = 1
        elif(self.options['L1freq']>0):
            n2use = 1
        else:
            n2use = 0
        #n2use is set to 0 if the ell_1 term is to be treated the same as the least-squares terms in that it will
        #be included in the greedy search.
        #otherwise, n2use is 1 if the ell_1 term is to be processed once every iteration or periodically, for
        # example once every 10 iterations

        # the three "outputs" are a sequence of function values, subgradient values, and the number of cumulative
        # Q-equivalent multiplies required to achieve those function values and subgradients (evaulated at x[findex])

        [fcaltimeT,chosen,since,pis,alphas] = self.initalizer(m,n_cols,n2use)
        rhoFlag = True
        rhoAct = np.zeros(m + 1, dtype='int')
        for k in range(self.options['maxIter']):


            mults2add = 0

            # greedy search for best terms to update
            [greedyi,sorted] = af.greedySearch(self.z[self.options['D']], self.w[self.options['D']], self.x, self.y,n2use)

            # Iks are which least-squares slices to update

            if( (not self.options['upperGreedy']) | (max(since)<self.options['M']) ):
                Iks = af.pickOperators(self.options['opSelectMethod'], m+1-n2use, self.options['randomOpAdd']
                                   , self.options['opPerIter'],k,greedyi,sorted)
                if(self.options['upperGreedy']):
                    sinceL = [since[i] + 1 for i in range(0, greedyi)]
                    sinceH = [since[i] + 1 for i in range(greedyi + 1, m+1-n2use)]
                    sinceL.append(0)
                    sinceL.extend(sinceH)
                    since = sinceL

            else:
                toupdate = np.array(since).argmax()
                Iks = af.pickOperators(self.options['opSelectMethod'], m, self.options['randomOpAdd']
                                   , self.options['opPerIter'], k, toupdate)
                sinceL = [since[i] + 1 for i in range(0, toupdate)]
                sinceH = [since[i] + 1 for i in range(toupdate + 1, m)]
                sinceL.append(0)
                sinceL.extend(sinceH)
                since = sinceL
                #chosen.append(toupdate)

            if(self.options['L1every']):
                # we always update the ell-term
                Ik = np.ones(1, 'int')
                Ik = np.concatenate((Ik, Iks))

            elif(k%self.options['L1freq']==0):
                Ik = np.ones(1, 'int')
                Ik = np.concatenate((Ik, Iks))
            elif(n2use==0):
                Ik = Iks
            else:
                Ik = np.zeros(1, 'int')
                Ik = np.concatenate((Ik, Iks))


            for i in range(m + 1):

                if(Ik[i]):
                    if( (self.options['D'] == 0) | ((i==0) & self.options['L1every'])):
                        # no delay for the proximal step as assumed it will be implemented by a central server or
                        # via MP reduction type operations.
                        z2use = self.z[self.options['D']]
                        wi2use = self.w[self.options['D']][i]
                    else:
                        #randomly generate between max(lastCall[i],k-D) and k
                        [delay_i,self.lastCall[i]] = af.generateDelay(self.options['D'], k, self.lastCall[i])

                        z2use = self.z[self.options['D'] - delay_i]
                        wi2use = self.w[self.options['D'] - delay_i][i]

                    if (i == 0):
                        # proximal step wrt ell_1 term

                        a = z2use + rho[i] * wi2use


                        self.x[i] = px.proxL1(a, rho[i], self.lam)
                        self.y[i] = rho[i] ** (-1) * (a - self.x[i])

                    else:
                        # update the i-th least-squares slice

                        if (self.options['doForward']):
                            # forward step
                            if(self.options['doBackTrack'] == False):
                                # no backtracking linesearch
                                Tiz = LSmod.gradLinear(z2use, self.Q[i-1], self.b[self.partition_list[i - 1]])


                                # simple fixed stepsize
                                self.x[i] = z2use - rho[i] * (Tiz - wi2use)

                                self.y[i] = LSmod.gradLinear(self.x[i], self.Q[i-1], self.b[self.partition_list[i - 1]])

                                # there have been four multiplies by matrix Q_i
                                mults2add += 4*len(self.partition_list[i-1])/float(n_rows)
                            else:
                                # backtracking linesearch
                                [self.x[i], self.y[i], rho[i]] =\
                                    self.lineSearchLS(z2use, wi2use,i , rho[i], self.options['Delta'],self.options['doAutoBT'])

                                if(self.options['equalizeRho']):
                                    # After backtracking, set the stepsize corresponding to the ell_1 norm to be the average
                                    # of the stepsizes across least-squares slices. This is not necessary for convergence
                                    # but has been found, on occasion, to improve empirical performance.
                                    rhoAct[i] = 1
                                    if (rhoFlag):
                                        rhosDone = []
                                        if (sum(rhoAct) == m):
                                            rhoFlag = False
                                            rhosDone = range(1, m + 1)
                                        else:
                                            for i in range(m + 1):
                                                if (rhoAct[i]):
                                                    rhosDone.append(i)

                                    else:
                                        rhosDone = range(1, m + 1)

                                    totRhoAct = sum(rhoAct)
                                    rho[0] = sum(np.array(rho)[np.array(rhosDone)])/(totRhoAct)


                                # 4 matrix multiplies by Q_i or Q_i^\top
                                # note that this is true even for the traditional backtracking procedure, since all the
                                # matrix multiplies are precomputed
                                mults2add += 4.0*len(self.partition_list[i-1])/float(n_rows)



                        else:
                            # a backward step on a least-squares slice
                            a = z2use + rho[i]*wi2use
                            # uses conjugage gradients initilized at the current value for x[i] for a max of maxInner iterations
                            # and checking the sigma-error criterion
                            [self.x[i],self.y[i],iterCG] = LSmod.proxQuad_cg(a, rho[i], self.Q[i-1], self.Atbis[i - 1],\
                                               self.options['maxInner'],self.options['doErrCheck'],self.options['sigma'],z2use,wi2use,self.x[i])
                            mults2add += iterCG * len(self.partition_list[i - 1])/float(n_rows)

            # Do the halfspace projection as in lines 13 to 27


            [znext, wnext]  = af.doProjection(self.x, self.y, m + 1, self.z[self.options['D']], self.w[self.options['D']],\
                                                self.options['gamma'], self.options['reducedW'], self.options['beta'])


            for d in range(self.options['D']):
                self.z[d] = self.z[d + 1]
                self.w[d] = self.w[d + 1]
            self.z[self.options['D']] = znext
            self.w[self.options['D']] = wnext

            self.multsAll[k+1]= self.multsAll[k] + mults2add

            if (k % m/self.options['opPerIter'] == 0):
                #only calculate the function values every m iterations as this can take some time and really slow down
                # the algorithm on large problems
                print"PS"+string+" iteration: "+str(k)
                if(self.options['funcIndex']>=0):
                    x2use = self.x[self.options['funcIndex']]
                else:
                    x2use = self.z[self.options['D']]

                fcaltime = time.time()
                self.getTerminationVal(x2use)
                fcaltime = time.time() - fcaltime
                fcaltimeT += fcaltime
                self.mults.append(self.multsAll[k + 1])
                self.iters.append(k+1)


        print"time to calculate functions:"+str(fcaltimeT)
        self.runtime = time.time() - t - fcaltimeT   #running time is adjusted by subtracting time taken to compute function values
        self.sparsity = sum(abs(self.x[self.options['funcIndex']])>1e-5)


    def createListofMats(self):
        # this function is purely for simplifying notation. Q[i] = A[partition_list[i],:]
        self.Q = []
        for i in range(len(self.partition_list)):
            self.Q.append(self.A[self.partition_list[i], :])




    def createDataStructs(self,m,n_cols):

        self.z = np.zeros([self.options['D'] + 1, n_cols])

        self.w = np.zeros([self.options['D'] + 1, m + 1,n_cols])
        self.lastCall = [-1 for i in range(m + 1)]

        self.x = np.zeros([m + 1,n_cols])
        self.y = np.zeros([m + 1,n_cols])


    def getAtbs(self,m):
        self.Atbis = []
        for i in range(m):
            self.Atbis.append(self.A[self.partition_list[i]].T.dot(self.b[self.partition_list[i]]))

    def lineSearchLS(self,z, wi, i, rhoi, Delta,doAuto):
        # Backtracking linesearch for the ith least-squares slice
        Ai = self.Q[i-1]
        bi = self.b[self.partition_list[i - 1]]
        Atbi = self.Atbis[i-1]


        zBG = Ai.T.dot(Ai.dot(z)) - Atbi
        rightPart = Ai.T.dot(Ai.dot(zBG - wi))

        if(doAuto==True):
            # Automatic rule for affine operators
            numerArg = zBG - wi
            numer = np.linalg.norm(numerArg)**2

            denomright = numerArg.dot(rightPart)

            denom = Delta*numer + denomright
            if (denom == 0):
                rho = 1.0
            else:
                rho = numer/denom

            rho = min(0.5*rho,rhoi)

            x = z - rho * numerArg
            y = zBG - rho * rightPart



        else:
            #original backtracking
            t = 1
            rho = rhoi
            while t>0:
                x = z - rho*(zBG - wi)
                y = zBG - rho*rightPart
                t = Delta*np.linalg.norm(z-x)**2 - (z-x).T.dot(y - wi)
                rho = 0.5*rho
            rho = 2*rho

        return [x,y,rho]


    def getOptStepFor(self,Bz,wi,Ai, Atbi):
        B_lz = Bz + Atbi
        denomRight = Ai.T.dot(Ai.dot(Bz - wi))
        rho = (np.linalg.norm(Bz - wi,2)**2)/(2*(Bz - wi).T.dot(denomRight))
        return [rho,denomRight]

    def getTerminationVal(self,x2use):
        if(self.options['funcVals']):
            f_2add = LSmod.L1LSvalue(self.A, self.b, x2use, self.lam)
            self.fout.append(f_2add)
        if(self.options['subgVals']):
            sg_2add = LSmod.maxSubg(self.A, self.b, self.lam, x2use)
            self.sgout.append(sg_2add)


    def initalizer(self,m,n_cols,n2use):
        self.fout = []
        self.multsAll = np.zeros(self.options['maxIter'] + 1)
        self.mults = []
        self.iters = []
        self.sgout = []
        fcaltimeT = 0.0
        chosen = []
        since = [0 for i in range(m+1-n2use)]
        pis = []
        alphas = []
        S = 0
        self.getTerminationVal(self.x[self.options['funcIndex']])
        self.mults.append(0)
        self.iters.append(0)

        return [fcaltimeT,chosen,since,pis,alphas]











