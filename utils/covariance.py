import logging
import numpy as np
from . import utils

class MockCovariance(object):

    logger = logging.getLogger('MockCovariance')

    def __init__(self,list_x,list_y):
        self._x = np.mean(list_x,axis=0)
        self._mean = np.mean(list_y,axis=0)
        self._covariance = np.cov(np.array(list_y).T,ddof=1)
        #self._std = np.std(list_y,axis=0,ddof=1)
        self.nobs = len(list_y)

    @classmethod
    def load_files(cls,reader,list_path=[]):
        list_x,list_y = [],[]
        for path in list_path:
            x,y = reader(path)
            list_x.append(x)
            list_y.append(y)
        return cls(list_x,list_y)

    def mean(self,x):
        return np.interp(x,self._x,self._mean)

    def std(self,x):
        #print('lol')
        #return np.interp(x,self._x,self._std)
        return np.interp(x,self._x,np.diag(self._covariance)**0.5)

    def getstate(self):
        state = self.__dict__
        return state

    def setstate(self,state):
        self.__dict__.update(state)

    def save(self,path):
        utils.mkdir(path)
        self.logger.info('Saving to: {}.'.format(path))
        np.save(path,self.getstate())

    @classmethod
    def load(cls,path):
        cls.logger.info('Loading: {}.'.format(path))
        self = object.__new__(cls)
        self.setstate(np.load(path,allow_pickle=True)[()])
        return self
