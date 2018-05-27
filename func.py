from net import Net
import visdom
from dataset import DataSet
from BP import BP

config1 = {
    
    'lr':0.0001,
    'max_epoch':500,
    'vis':visdom.Visdom(env=u'plus_loss_test'),
    'net':Net(2,[5,3,1],0.0001,'relu'),
    'data':DataSet('x1+x2')
        
 }

config2 = {
    
    'lr':0.01,
    'max_epoch':1000,
    'vis':visdom.Visdom(env=u'sin_loss_test'),
    'net':Net(1,[5,4,3,1],0.01,'relu'),
    'data':DataSet('sinx')
        
 }

config3 = {
    
    'lr':0.01,
    'max_epoch':1000,
    'vis':visdom.Visdom(env=u'x2_loss_test'),
    'net':Net(1,[20,1],0.01,'relu'),
    'data':DataSet('x2')
        
 }
process = BP(config=config2)
process.run()