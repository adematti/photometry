import numpy as np

class BaseModel(object):

    def __init__(self,density):
        self.data = density.brickdata
        self.randoms = density.brickrandoms
        self.properties = density.properties

    @property
    def size(self):
        return len(self.randoms)

class LinearModel(BaseModel):

    def templates(self,catalogue=None):
        if catalogue is None: catalogue = self.properties
        return np.array([np.ones(self.size)]+[catalogue[prop] for prop in self.props]).T

    def fit(self,props=[]):
        self.props = props
        randoms = self.randoms*self.data.sum()/self.randoms.sum()
        templates = randoms[:,None]*self.templates()
        precision = 1/randoms
        xfx = (templates.T*precision).dot(templates)
        xfy = np.sum(templates.T*precision*self.data,axis=-1)
        self.coeffs = np.linalg.inv(xfx).dot(xfy)

    def predict(self,catalogue=None,key_efficiency='LINEFF'):
        toret = np.sum(self.coeffs*self.templates(catalogue=catalogue),axis=-1)
        if catalogue is not None and key_efficiency is not None:
            catalogue[key_efficiency] = toret
        return toret
