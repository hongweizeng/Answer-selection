import math
import torch.nn as nn

class Criterion(object):
    
    def __init__(self):
    	super(Criterion, self).__init__()


    def loss(scores, lables, generator, crit, eval=False):
        # compute generations one piece at a time
        num_correct, loss = 0, 0
        scores = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
        batch_size = outputs.size(1)
        
        loss = torch.dot(torch.log(scores), labels) + torch.dot(torch.log(1-scores), (1-labels))

        loss_data = loss.data[0]
        if not eval:
            loss.div(batch_size).backward()

        grad_output = None if scores.grad is None else scores.grad.data

        return loss, grad_output
