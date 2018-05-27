import numpy as np
class BP:

    def __init__(self,config):
        self.lr = config['lr']
        self.max_epoch = config['max_epoch']
        self.vis = config['vis']
        self.net = config['net']
        self.data = config['data']
        self.loss_p = []
        self.loss_num = []
    
    def run(self):
        
        for epoch in range(self.max_epoch):
            self.train(epoch)
            self.val_test_all(self.data.val(),name='val_fitting')
            self.val_test_all(self.data.test(),name='test_fitting')
            self.val_test_all(self.data.all_data(),name='all_fitting')
        
    def train(self,epoch):
        
        x,y = self.data.train()
        loss_all = 0.0
        for i in range(len(x)):
            loss,result = self.net.forward(x[i],y[i])
            self.net.backward()
            loss_all += loss
        print("The {} epoch, Loss is {}".format(epoch,loss_all/len(x)))
        self.loss_p.append(epoch)
        self.loss_num.append(loss_all/len(x))
        self.vis.line(X=np.array(self.loss_p),
                      Y=np.array(self.loss_num),
                      win='train_loss')
        
    def val_test_all(self,data,name):     
        
        x,y = data
        scatter = []
        label = []
        for i in range(len(x)):
            loss,result = self.net.forward(x[i],y[i])
            scatter,label = self.updateScatter(x[i],y[i],result,scatter,label)
        self.drawScatter(scatter,label,name=name)
    
            
    def updateScatter(self,x,y,result,scatter,label):
        
        scatter.append(np.append(x,y))
        scatter.append(np.append(x,result))
        label.append(1)
        label.append(2)
        return scatter,label
    
    def drawScatter(self,scatter,label,name=None):

        self.vis.scatter(
            X=np.array(scatter),
            Y=np.array(label),
            opts=dict(
                markersize=10,
                legend=['true_'+name, 'result_'+name],
            ),
            win=name
        )
        
    
        
        