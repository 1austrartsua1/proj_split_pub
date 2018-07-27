# Copyright 2018, Patrick R. Johnstone
#    This toolbox is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License <http://www.gnu.org/licenses/> for more
#    details.


import numpy as np
import proxes as px
import time


def sign(xi):
    return (2.0*(xi>0)-1.0)

def thresh(a):
    if(a>1.0):
        return 1.0
    elif(a<-1.0):
        return -1.0
    else:
        return a

def threshVec(vec,x,lam):
    tolerance = 1e-10
    threshed = lam*(vec <= -lam)-lam* (vec>= lam)
    inside = -vec*(vec>-lam)*(vec<lam)
    suppxc = 1.0*(abs(x)<tolerance)
    sgnx = (1-suppxc)*(2.0*(x>0)-1)*lam
    return suppxc*(threshed + inside) + sgnx

def maxSubg(A,b,lam,x):
    gradL2 = gradLinear(x, A, b)
    gradL1 = threshVec(gradL2,x,lam)
    #gradL1 = np.zeros(len(x))
    #for i in range(len(x)):
    #    if(x[i]==0):
    #        temp = -lam**(-1)*gradL2[i]
    #        gradL1[i] = lam*thresh(temp)
    #    else:
    #        gradL1[i]=sign(x[i])*lam
    return np.linalg.norm(gradL2+gradL1)

def gradLinear(z,Q,v):
    return Q.T.dot(Q.dot(z)-v)

def L1LSvalue(A,b,z,lam):
    return 0.5*np.linalg.norm(A.dot(z)-b)**2+lam*np.linalg.norm(z,1)

def FISTAL1LS(A,b,lam,x0,tau,maxIter,elasNet = False,lam2 = -1,backTrack = False,q=0.0,doSubgs=False):
    t = 1.0
    theta = 1.0
    beta = 0.0
    x = x0
    xold = x0
    y = x0
    f = [L1LSvalue(A, b, x, lam)]
    subgs = [maxSubg(A, b, lam, x)]
    iterFista = [0]
    tsubgs = 0
    for k in range(maxIter):
        if(k%10==0):
            print"Fista iteration: " + str(k)

        grady = gradLinear(y, A, b)

        if(backTrack):
            fnew = float('inf')
            tau = 2 * tau
            Qy = 0
            i = 0
            Ay = A.dot(y)
            while (fnew > Qy):
                tau = 0.5*tau
                xnew = px.proxL1(y - tau * grady, tau, lam)

                fnew = L1LSvalue(A, b, xnew, lam)
                Qy = 0.5 * np.linalg.norm(Ay - b, 2) ** 2 + grady.T.dot(xnew - y) \
                         + 0.5 * tau ** (-1) * np.linalg.norm(xnew - y) ** 2 + lam * np.linalg.norm(xnew, 1)
                i +=1

        else:
            xnew = px.proxL1(y - tau * grady, tau, lam)
            fnew = L1LSvalue(A, b, xnew, lam)
            i = 0

        f.append(fnew)
        if(doSubgs):
            tnewSubg = time.time()
            subgs.append(maxSubg(A, b, lam, xnew))
            if(k>500):
                if(subgs[len(subgs)-1]>10*subgs[len(subgs)-2]):
                    print"debugging needed"
                    print"iteration: " + str(k)

            tnewSubg = time.time() - tnewSubg
            tsubgs += tnewSubg

        iterFista.append(iterFista[len(iterFista)-1]+2+i)
        xoldold = xold
        xold = x
        x = xnew
        if(q == 0):
            tnew = 0.5 + 0.5 * np.sqrt(1 + 4 * t**2)
            beta = (t - 1) / tnew
            t = tnew
        elif(q==-1.0):
            #running prox grad
            beta = 0.0
        else:
            thetanew = (q-theta**2 + np.sqrt((q-theta**2)**2 + 4*theta**2))/2
            if(thetanew > 1.0):
                thetanew = (q - theta ** 2 - np.sqrt((q - theta ** 2) ** 2 + 4 * theta ** 2)) / 2

            beta = (theta*(1-theta))/(theta**2 + thetanew)
            theta = thetanew

        y = x + beta * (x - xold)

    return [f,x,iterFista,subgs,tsubgs]


def proxGradL1LS(A,b,lam,x0,tau,maxIter,backtrackType,subgVal):
    [m,d]=A.shape
    xold=x0
    f_pg = [L1LSvalue(A, b, xold, lam)]
    gold = 0.5*np.linalg.norm(A.dot(x0)-b,2)**2
    fold = gold + lam*np.linalg.norm(x0,1)
    trueIters = 0
    numMults = [0]

    pg_subg =  [maxSubg(A,b,lam,xold)]
    tsubgvals = 0
    for k in range(maxIter):
        print"PG iteration: "+str(k)
        gnew = float('inf')
        gradxOld = gradLinear(xold, A, b)
        innerIters = 0
        if(backtrackType == 'backTrack'):
            tau = 2 * tau
            G = np.zeros(d)
            while (gnew > (gold - tau*gradxOld.T.dot(G) + 0.5*tau*np.linalg.norm(G,2)**2)):
                tau = 0.5 * tau
                x = px.proxL1(xold-tau*gradxOld,tau,lam)
                G = tau**(-1)*(xold - x)
                innerIters += 1
                gnew = 0.5*np.linalg.norm(A.dot(x)-b,2)**2
                fnew = gnew + lam*np.linalg.norm(x,1)

            numMults.append(numMults[len(numMults)-1] + 2 + innerIters)
        else:
            x = px.proxL1(xold - tau * gradxOld, tau, lam)

            fnew = L1LSvalue(A, b, x, lam)
            numMults.append(numMults[len(numMults)-1] + 2)



        fold = fnew
        gold = gnew
        xold = x
        f_pg.append(fold)

        if(subgVal):
            tsubgval = time.time()
            pg_subg.append(maxSubg(A,b,lam,x))
            tsubgval = time.time() - tsubgval
            tsubgvals += tsubgval
    print"prox grad sparsity: "+str(sum(abs(x)>1e-5))

    return [x,f_pg,numMults,pg_subg,tsubgvals]

def proxQuad_cg(a,rho,Q,Qtv,maxInner,errCheck,sigma,z,w,x0):
    # computes the prox of the least-squares term wrt to Q and  v
	# via conjugate gradients. While checking the relative error criterion
	# Computes approximately \min (rho/2)*\|Qx - v\|^2 + (1/2)*\|x-a\|^2
	# This boils down to a system of linear equations

    #A = np.eye(dim)+rho*QtQ
    b = a+rho*Qtv
    x = x0
    y = Q.T.dot(Q.dot(x)) - Qtv
    r = b - Acg(x,Q,rho)
    p = r
    for i in range(maxInner):
        if(np.linalg.norm(p)<1e-10):
            return [x, y, 4 * (i+1)]
        Ap = Acg(p,Q,rho)
        alpha = r.T.dot(r)/(p.T.dot(Ap))
        x = x + alpha*p
        rnew = r - alpha*Ap
        beta = rnew.T.dot(rnew)/(r.T.dot(r))
        p = rnew+beta*p
        r = rnew
        if(errCheck):
            QtQx = Q.T.dot(Q.dot(x))
            y = QtQx-Qtv
            e = x+rho*y - a
            if( (z-x).T.dot(e) >= -sigma*(z-x).T.dot(z-x) ):
                if( (y-w).T.dot(e) <= rho*sigma*(y-w).T.dot(y-w)):
                    #error check satisfied
                    y = QtQx - Qtv
                    return [x,y,4*(i+2)]

    return [x,y,4*(MaxInner+1)]

def Acg(x,Q,rho):
    return (x + rho*Q.T.dot(Q.dot(x)))
