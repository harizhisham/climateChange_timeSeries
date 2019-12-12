# build ARMA(p,q) process
# phis = [phi_1, ..., phi_p] # a list of phi values
# thetas = [theta_1, ..., theta_q]
import numpy as np

def build_arma(phis, thetas, errors, constant=0, show=False):
    Y = [constant] * max([len(phis), len(thetas)])
    n = len(phis)
    i = len(phis)
    for e in errors[n:]:
        value = constant
        for lag in range(len(phis)):
            value += phis[lag] * Y[-lag - 1]
        for lag in range(len(thetas)):
            value += thetas[lag] * errors[i - lag - 1]
        y = value + e
        Y += [y]
        i += 1
            
    if show:
        print('| ARMA(%d, %d) process | mean: %.4f | st_dev: %.4f |' % (len(phis),len(thetas), np.mean(Y), np.std(Y)))
        print('---')
        for n,phi in enumerate(phis):
            print('phi_%d = %+.4f' % (n+1, phi))
        print('---')
        for n,theta in enumerate(thetas):
            print('theta_%d = %+.4f' % (n+1, theta))
        print('---')
    return Y