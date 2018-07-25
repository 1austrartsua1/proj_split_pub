import numpy as np

def greedySearch(z,w,x,y,n):
    # the greedy search is only over the Lipschitz terms, i.e. NOT the proximal terms, as these are tackled every
    # iteration in our methods as implemented here
    # n is the number of proximal terms
    [npm,_] = x.shape

    #phimx = float('inf')
    #j = n
    #argmins = []
    #for i in range(n,npm):
    #    phii = (z - x[i]).T.dot(y[i]-w[i])
    #    if(phii==phimx):
    #        phimx = phii
    #        argmins.append(i)
    #    elif(phii<phimx):
    #        phimx = phii
    #        argmins = [i]


    #j = int(np.random.choice(argmins,1))

    phi = np.array([(z-x[i]).T.dot(y[i]-w[i]) for i in range(n,npm)])
    sorted = phi.argsort()
    j = sorted[0]+n

    return [(j-n),sorted]



def pickOperators(opSelectMethod,tot,randomOpAdd,opPerIter,k,greedyi = -1,sorted=[]):
    Ik = np.zeros(tot, 'int')
    if (randomOpAdd):
        ops2add = np.random.choice(opPerIter) + 1
    else:
        ops2add = opPerIter

    if (opSelectMethod == 'Random'):
        Toadd = np.random.choice(range(tot), [ops2add, ], False)
        Ik[Toadd] = 1
    elif (opSelectMethod == 'Cyclic'):
        Toadd = k % (tot)
        Ik[Toadd] = 1
    elif (opSelectMethod == 'Greedy'):
        #Ik[greedyi] = 1
        Ik[sorted[0:ops2add]]=1
    else:
        Ik = np.ones(tot,'int')
    return Ik

def generateDelay(D,k,lastCall_i):
    delay_i = np.random.choice(D + 1)
    lagIter = k - delay_i
    if (lagIter <= lastCall_i):
        lagIter = lastCall_i
    delay_i = k - lagIter
    lastCall_i = lagIter
    return [delay_i,lastCall_i]



def doProjection(x,y,n_parts,z,w,gamma = 1.0,reducedW = False,beta = 1.0):
    # projection steps on lines 13 through 27

    v = np.sum(y,0)


	# reducedW controls whether we use the hyplerplane as defined in the paper
	# or as defined in Eckstein (2017): "A Simplified Form of Block-Iterative Operator Splitting, 
	# and an Asynchronous Algorithm Resembling the Multi-Block ADMM"
    # When reducedW is True we are using the hyplerplane from our paper
	# When reduced W is false, we use the slightly different hyperplane from Eckstein (2017)

    if(reducedW==False):
        xbar = np.sum(x,0) / n_parts
    else:
        xbar = x[n_parts-1]



    u = x -  xbar

    
    pi = np.linalg.norm(u) ** 2 + gamma**(-1)*np.linalg.norm(v) ** 2

    
    if (pi > 0):
        term1 = z.dot(v)

        term2 = sum([x[i].dot(w[i]-y[i]) for i in range(n_parts)])

        hplane = term1 + term2
    
        alpha = (pi ** (-1)) * max([0, hplane])

    else:
        alpha = 0

    znew = z - gamma**(-1)*beta * alpha * v


    wnew = np.subtract(w,beta * alpha * u)

    if(reducedW):
        wnew[n_parts-1]=-sum(wnew[0:(n_parts-1)])


    return [znew,wnew]



