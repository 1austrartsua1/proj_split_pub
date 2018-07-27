# Copyright 2018, Patrick R. Johnstone
#    This toolbox is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License <http://www.gnu.org/licenses/> for more
#    details.

import numpy as np


def proxL1(a,rho,lam):
    rholam = rho * lam
    out = (a> rholam)*(a-rholam)
    out+= (a<-rholam)*(a+rholam)
    #absa = np.abs(a+1e-15)
    #out = a / np.abs(a+1e-15) * np.maximum(np.abs(a) - rholam, 0)
    #out = pywt.threshold(a,rholam)

    return out










