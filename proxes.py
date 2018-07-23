import numpy as np


def proxL1(a,rho,lam):
    rholam = rho * lam
    out = (a> rholam)*(a-rholam)
    out+= (a<-rholam)*(a+rholam)
    #absa = np.abs(a+1e-15)
    #out = a / np.abs(a+1e-15) * np.maximum(np.abs(a) - rholam, 0)
    #out = pywt.threshold(a,rholam)

    return out










