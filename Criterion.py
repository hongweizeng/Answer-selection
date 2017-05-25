import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Criterion(object):
    
    def __init__(self, opt):
    	super(Criterion, self).__init__()

        self.crit = nn.NLLLoss(size_average=False)
        if opt.gpus:
            self.crit.cuda()

    def loss(self, scores, labels, generator, crit, eval=False):
        # compute generations one piece at a time
        num_correct, loss = 0, 0
        scores = Variable(scores.data, requires_grad=(not eval), volatile=eval)

        scores = generator(scores)
        print scores
        print labels
        batch_size = scores.size(0)
        
        loss = torch.dot(torch.log(scores), labels) + torch.dot(torch.log(1-scores), (1-labels))

        loss_data = loss.data[0]
        if not eval:
            loss.div(batch_size).backward()

        grad_output = None if scores.grad is None else scores.grad.data

        return loss, grad_output
