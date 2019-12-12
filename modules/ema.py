'''
Created on Sep 22, 2019

@author: dan trepanier
'''
import numpy as np
import matplotlib.pyplot as plt
import prettytable

import plumbing

'''
This class is a wrapper to calculate RMSE
This facilitates optimization of parameters

Arguments
---------
data            = time series data
avg_constructor = class constructor for the moving average
lookahead       = lookahead overwhich to minimize error
'''
class Runner(object):
    def __init__(self, title, data, avg_constructor, lookahead=1):
        self.title = title
        self.data = data
        self.avg_constructor = avg_constructor
        self.lookahead = lookahead
        
    def create_instance(self, args):
        return self.avg_constructor(*args)
    
    def run(self, args):
        A = []
        avg_class = self.create_instance(args)
        for x in self.data:
            A += [avg_class.post(x)]
        return A
    
    def get_residuals(self, args):
        R = []
        avg_class = self.create_instance(args)
        for i in range(len(self.data) - self.lookahead):
            x = self.data[i]
            avg = avg_class.post(x)
            x_hat = avg_class.predict(self.lookahead)
            if None not in (x,x_hat):
                R += [(self.data[i + self.lookahead] - x_hat)] 
        return R
    
    def get_rms_error(self, args):
        R = self.get_residuals(args)
        E_square = list(map(lambda r: r ** 2, R))
        return np.mean(E_square) ** .5
    
    def sweep(self, values, show=True):
        E = []
        pt = prettytable.PrettyTable(['value','RMSE'])
        for v in values:
            e = self.get_rms_error(v)
            E += [e]
            pt.add_row([v,e])
        if show:
            plt.plot(E)
            plt.title('%s Sweep' % self.title)
            plt.show()
            print(pt)
        return E
    
class SimpleMovingAverage(object):
    def __init__(self, lookback=None):
        self.lookback = lookback
        self.fifo = plumbing.FIFO(lookback)
    
    def post(self, x):
        buf = self.fifo.post(x)
        if len(buf) == self.lookback or self.lookback is None:
            return np.mean(buf)
        else:
            return None

    def predict(self, n):
        buf = self.fifo.data
        if len(buf) > 0:
            return np.mean(buf)
        else:
            return None

class ExponentialMovingAverage(object):
    def __init__(self, alpha, ema=None):
        assert alpha > 0.0 and alpha < 1.0,'unexpected alpha value: %f' % alpha
        self.alpha = alpha
        self.ema = ema
    
    def post(self, x):
        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.alpha * x + (1 - self.alpha) * self.ema
        return self.ema
    
    def predict(self, n):
        return self.ema


'''
Holt's Method of Double Exponential Smoothing

alpha   : alpha value between 0 and 1
beta    : beta value between 0 and 1
s       : s_0 value to initialize the EMA -- if None, s_0 = x_0
b       : b_0 value to initialize the slope component (b) -- if None, b_0 = x_1 - x_0
'''

class DoubleEMA(object):
    def __init__(self, alpha, beta, s=None, b=None):
        self.alpha = alpha
        self.beta = beta
        self.s = s
        self.b = b
        self.last = None
    
    def post(self, x):
        if self.s is None:
            self.s = x
        elif self.b is None:
            self.s = self.alpha * x + (1 - self.alpha) * self.s
            self.b = x - self.last
        else:
            s = self.alpha * x + (1 - self.alpha) * (self.s + self.b)
            self.b = self.beta * (s - self.s) + (1 - self.beta) * self.b
            
            self.s = s
        self.last = x
        return self.s
    
    def predict(self, n):
        if self.s and self.b:
            return self.s + n * self.b
        else:
            return self.s