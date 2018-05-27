import math
class Relu:
    def __init__(self):
        pass
    
    def forward(self,x):
        x = x[0]
        if x>0:
            return x
        else:
            return 0.5*x
        
    def  derivate(self,x):
        x = x[0]
        if(x>0):
            return 1
        else:
            return 0.5  
    
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self,x):
        x = x[0]
        return 1/(1+math.exp(-x))
    
    def derivate(self,x):
        x = x[0]
        return x*(1-x)

class Tanh:
    def __init__(self):
        pass
    
    def forward(self,x):
        x = x[0]
        return math.tanh(x)
    
    def derivate(self,x):
        x = x[0]
        return 1-x*x
           
    