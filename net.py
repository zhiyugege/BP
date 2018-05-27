import numpy as np
from activate import Relu,Sigmoid,Tanh
class Net:
    def __init__(self,dim,layers,lr,activate=None):
        self.inputs = None
        self.label = None
        self.layers = [dim]+[layer for layer in layers]
        self.lr = lr
        self.weights,self.bias,self.outs,self.pre_outs,self.gradient = self.init_weights()
        self.act_name = activate        
        self.act = self.init_act_func(activate)
    
    def forward(self,inputs,label):
        
        self.inputs = inputs
        self.label = label
        self.outs = [np.array(self.inputs)]+[np.zeros((self.layers[i],1)) for i in range(1,len(self.layers))]
        x = self.inputs
        for i in range(1,len(self.layers)):
            self.outs[i] = np.dot(self.weights[i-1],x)+np.transpose([np.sum(np.array(self.bias[i-1]), axis=1)])
            self.pre_outs[i] = (self.outs[i]).copy()
            self.outs[i] = np.array([[self.act.forward(out)] for out in self.outs[i]])
            x = (self.outs[i]).copy()
        return self.loss(),self.outs[-1][0]

    
    def backward(self):
        
        self.getGradient()
        lr = self.lr        
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = [weight-lr*self.outs[i][k][0]*self.gradient[i+1][j] for k,weight in enumerate(self.weights[i][j])]
                self.bias[i][j] = [bias-lr*self.gradient[i+1][j] for bias in self.bias[i][j]]
                
    def getGradient(self):
        
        out = (self.outs[-1][0]).copy()
        pre_out = (self.outs[-1][0]).copy()
        if self.act_name=='sigmoid' or self.act_name=='tanh':
            self.gradient[-1] = [(out[0]-self.label[0])*self.act.derivate(out)]
        else:
            self.gradient[-1] = [(out[0]-self.label[0])*self.act.derivate(pre_out)]
        for i in range(len(self.layers)-2,0,-1):
            for j in range(len(self.gradient[i])):
                self.cal(i,j)
        
    def cal(self,i,j):
        weight = np.transpose(self.weights[i])
        weight = (weight[j]).copy()
        gradient = (self.gradient[i+1]).copy()
        if self.act_name=='sigmoid' or self.act_name=='tanh':
            self.gradient[i][j] = np.dot(np.transpose(weight),gradient)*self.act.derivate(self.outs[i][j])
        else:
            self.gradient[i][j] = np.dot(np.transpose(weight),gradient)*self.act.derivate(self.pre_outs[i][j])
            
    def loss(self):
        
        return (self.outs[-1][0][0]-self.label[0])**2/2.0
    
    def init_weights(self):
        layers = self.layers
        new_weights = [np.random.rand(layers[i+1],layers[i]) for i in range(len(layers)-1)]
        new_bias = [np.random.rand(layers[i+1],layers[i]) for i in range(len(layers)-1)]
        outs = [np.array(self.inputs)]+[np.zeros((layers[i],1)) for i in range(1,len(layers))]
        pre_outs = [np.array(self.inputs)]+[np.zeros((layers[i],1)) for i in range(1,len(layers))]
        gradient = [np.zeros(layers[i]) for i in range(0,len(layers))]
        return new_weights,new_bias,outs,pre_outs,gradient
    
    def init_act_func(self,function):
        
        if function==None or function=='relu':
            return Relu()
        if function=='sigmoid':
            return Sigmoid()
        if function=='tanh':
            return Tanh()

#net = Net(1,[3,2,1],0.01,'sigmoid')
#net.forward([[[1]]],[[2]])