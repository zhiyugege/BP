import numpy as np
import math

class DataSet:
    
    def __init__(self, name=None):
        self.name = name
        self.x = []
        self.y = []
        self.getData()
    
    def all_data(self):
        
        return self.x,self.y
    
    def train(self):
        
        x = self.x
        y = self.y
        return x[0:int(0.5*len(x))],y[0:int(0.5*len(y))]
    
    def val(self):
        
        x = self.x
        y = self.y
        return x[0:int(0.25*len(x))],y[0:int(0.25*len(y))]
    
    def test(self):
        x = self.x
        y = self.y
        return x[int(0.5*len(x)):],y[int(0.5*len(y)):]
    
    def getData(self):
        
        x = []
        y = []
        if self.name=='x1+x2':
            for i in range(1000):
                a = np.random.randint(-5,5)
                b = np.random.randint(-5,5)
                x.append([[a],[b]])
                y.append([a+b])
        elif self.name=='sinx':
            for i in range(1000):
                a = np.random.randint(-314,314)*0.01
                x.append([[a]])
                y.append([math.sin(a)])
        elif self.name=='x2':
            for i in range(500):
                a = np.random.randint(-300,300)*0.01
                x.append([[a]])
                y.append([a*a])
        elif self.name =='cycle':
            for i in range(1000):
                a = np.random.randint(-5,5)
                b = np.random.randint(-5,5)
                x.append([[a],[b]])
                x.append([[a],[b]])
                y.append([math.sqrt(a*a+b*b)])
                y.append([-math.sqrt(a*a+b*b)])
        
        self.x = x
        self.y = y
        