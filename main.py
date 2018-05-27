from net import Net
import visdom
from dataset import DataSet
from BP import BP

config = {
    
    'lr':0.0001,
    'max_epoch':500,
    'vis':visdom.Visdom(env=u'plus_loss_test'),
    'net':Net(2,[5,3,1],0.0001,'relu'),
    'data':DataSet('x1+x2')
        
 }

process = BP(config=config)
process.run()