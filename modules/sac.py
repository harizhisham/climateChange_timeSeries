'''
Created on Sep 17, 2019

@author: dan
'''
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')
import argparse
import numpy as np
import prettytable
import matplotlib.pyplot as plt


class Location(object):
    def __init__(self, longitude, latitude):
        self.loc = (longitude, latitude)
        self.data = {}
        self.X = None
    
    def post(self, date, metric):
        if date in self.data:
            logging.warning('duplicate data: %s %s' % (date, metric))
        self.data[date] = metric
    
    def get(self, index=None):
        if self.X is None:
            X = []
            for d in sorted(self.data.keys()):
                X += [self.data[d]]
            self.X = np.array(X)
        if index is None:
            return self.X
        else:
            return np.array([self.X[index]])

class Result(object):
    def __init__(self, d, i, j):
        self.d = d
        self.i = i
        self.j = j
        if d == 0:
            self.w_ij = 0
        else:
            self.w_ij = 1 / d ** .5
        self.variances = 0
        self.n = 0
        
    def post(self, var):
        self.variances += sum(var)
        self.n += len(var)
        
    def get(self):
        return self.variances / self.n 
    
    def get_weight(self):
        return self.w_ij

class Moran(object):
    def __init__(self, metric, x_label='longitude', y_label='latitude'):
        self.metric = metric
        self.x_label = x_label
        self.y_label = y_label
        self.locations = {}
        self.results = {}
        self.W = None # weight matrix
        self.I = None # Moran's I value
        self.E = None # Expected Value
        self.var = None # Expected Variance
        self.Z = None # Z score of measurement
        
    def post(self, x):
        k = (x[self.x_label],x[self.y_label])
        if k not in self.locations:
            self.locations[k] = Location(*k)
        self.locations[k].post(x.get('date',None), x[self.metric])
    
    def _get_x_bar(self, index):
        X = np.array([])
        for l in self.locations.values():
            X = np.append(X, l.get(index))
        return np.mean(X)

    def _post_result(self, l_1, l_2, i, j, metric):
        h = l_2.loc[0] - l_1.loc[0]
        v = l_2.loc[1] - l_1.loc[1]
        d = (h ** 2 + v ** 2) ** .5 # distance
        if d not in self.results:
            self.results[d] = Result(d, i, j)
        self.results[d].post(metric)
        return self.results[d]
    
    def _get_var(self, index, T):
        x_bar = self._get_x_bar(index)
        S_w = np.sum(self.W) * T
        N = len(self.locations) * T
        # variance
        s_1 = 0
        s_2 = 0
        
        num_3 = 0
        denom_3 = 0
        
        for i,l in enumerate(self.locations.values()):
            w_2_left = 0
            w_2_right = 0
            for j in range(len(self.locations)):
                s_1 += (self.W[i,j] + self.W[j,i]) ** 2
                w_2_left += self.W[i,j]
                w_2_right += self.W[j,i]
            s_2 += (w_2_left + w_2_right) ** 2
            X_i = l.get(index)
            num_3 += sum( (X_i - x_bar) ** 4 )
            denom_3 += (X_i - x_bar) ** 2 
            
        s_1 *= .5
        s_3 = 1/(N*T) * num_3 / (1/(N*T) * sum(denom_3)) ** 2
        s_4 = (N ** 2 - 3 * N +3) * s_1 - N * s_2 + 3 * S_w ** 2
        s_5 = (N **2 - N) * s_1 - 2 * N * s_2 + 6 * S_w ** 2
        for n,s in enumerate([s_1,s_2, s_3, s_4, s_5]):
            logging.debug('s_%d = %f' % (n+1, s))
        return (N * s_4 - s_3 * s_5) / ( (N-1) * (N-2) * (N-3) * S_w ** 2) - self.E ** 2
    
    def get(self, index=None, show=True):
        x_bar = self._get_x_bar(index)
        self.results = {}
        W = []
        numerator, denominator = 0,0
        S_w = 0
        n = 0
        T = None
        for i,l_1 in enumerate(self.locations.values()):
            logging.info('%d / %d' %(i+1, len(self.locations)))
            X_i = l_1.get(index)
            w_row = []
            if T is None:
                T = len(X_i) # number of time series
            else:
                assert T == len(X_i),'mismatch in length of time series! expected %d, got %d' % (T, len(X_i))
            
            for j,l_2 in enumerate(self.locations.values()):
                X_j = l_2.get(index)
                semi_var = (X_i - X_j) ** 2
                r = self._post_result(l_1, l_2, i, j, semi_var)
                w_ij = r.get_weight()
                numerator += w_ij * sum( (X_i - x_bar) * (X_j - x_bar) )
                S_w += w_ij * T
                w_row += [w_ij]
            W += [w_row]
            denominator += sum((X_i - x_bar) ** 2)
            n += T
        self.W = np.array(W)
        self.I = n/S_w * numerator/denominator
        self.E = -1 / (n - 1)
        self.var = self._get_var(index, T)
        self.Z = (self.I - self.E) / self.var ** .5
        logging.info('Metric=%s | Index=%s | found %d distinct distances' % (self.metric, index, len(self.results)))
        logging.debug('W matrix shape: %s' % str(self.W.shape))
        logging.debug('N: %d | T: %d' % (n, T))
        logging.debug('Moran I |   numerator=%f | S_w=%f | numerator/S_w: %f' % (numerator, S_w, numerator / S_w))
        logging.debug('Moran I | denominator=%f |   n=%f | denominator/n: %f' % (denominator, n, denominator / n))
        logging.debug('Moran I |      actual=%f | expected=%f | s.d.=%f | Z: %f' % (self.I, self.E, self.var**.5, self.Z))
        if show:
            self.plot(index)
        return {'I': self.I, 'E': self.E, 'var': self.var, 'Z': self.Z}
     
    def plot(self, index):
        X,Y,Z = [],[],[] # heatmap chart data
        for l in self.locations.values():
            X_i = l.get(index)
            X += [l.loc[0]]
            Y += [l.loc[1]]
            Z += [np.mean(X_i)]
        
        D = []
        V = []
        for d in sorted(self.results.keys()):
            r = self.results[d]
            D += [d]
            V += [r.get()]
        title ='Square Difference %s Time Series\nvs.\nDistance in Degrees' % (self.metric)
        
        left, width = 0.15, 0.8
        rects = [(left, 0.50, width, 0.35), # region plot -- top
                 (left, 0.10, width, 0.35),] # location plot -- bottom
                 
        fig = plt.figure(facecolor='white')
        axescolor = '#f6f6f6'
        ax = {}
        for n,rect in enumerate(rects):
            ax[n] = fig.add_axes(rect, facecolor=axescolor)
        
        fig.suptitle(title)
        low,high = min(Z),max(Z)
        s = ax[0].scatter(X,Y, c=Z, cmap='seismic',vmin=low,vmax=high)
        ax[1].scatter(D,V)
        plt.ylabel('%s^2' % self.metric)
        plt.xlabel('Distance')
        plt.show()

def main(metric, index, latitudes=[35,40],longitudes=[-100,-95]):
    assert metric in ['precip','wind','min_temp','max_temp']
    assert len(latitudes) == 2
    assert len(longitudes) == 2
    assert np.diff(latitudes) > 0
    assert np.diff(longitudes) > 0
    args = tuple([metric] + latitudes + longitudes)
    sql_stmt = "select date,latitude,longitude,%s from weather where latitude>=%f and latitude<=%f and longitude>=%f and longitude<=%f group by date,latitude,longitude order by date;" % args
    w = weather.Weather()
    raw = w.query(sql_stmt, ['date', 'latitude', 'longitude', metric])
    m = Moran(metric)
    
    for x in raw:
        m.post(x)
    result = m.get(index)
    print('-----\nMORAN\n-----')
    pt = prettytable.PrettyTable(['metric','vaue'])
    for x in result.items():
        pt.add_row(x)
    print(pt) 
    return result
    
if __name__ == "__main__":
    from .db import weather
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric','-m', help="metric", choices=['precip','wind','min_temp','max_temp'], type=str,default='wind')
    parser.add_argument('--index', help="if Index is None it tests the whole time series", type=int, default=None)
    args = parser.parse_args()
    logging.info('start')
    logging.info(args)
    main(args.metric, args.index)
    logging.info('done')