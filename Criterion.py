from __future__ import division

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Criterion(object):
    
    def __init__(self, model, opt, weights=[0.08, 0.92], factor=0.0005):
    	super(Criterion, self).__init__()
        weights = torch.FloatTensor(weights)
        self.crit = nn.NLLLoss(weight=weights, size_average=False)
        if opt.gpus:
            self.crit.cuda()
        self.l1_crit = nn.L1Loss(size_average=False)

        self.model = model
        self.factor = factor

    def loss(self, scores, labels, generator, eval=False):
        # compute generations one piece at a time
        num_correct, loss = 0, 0
        scores = Variable(scores.data, requires_grad=(not eval), volatile=eval)

        grad_output = None if scores.grad is None else scores.grad.data

        scores = generator(scores)
        batch_size = scores.size(0)
        labels = labels.view(-1)
        # loss = torch.dot(torch.log(scores), labels) + torch.dot(torch.log(1-scores), (1-labels))
        loss = self.crit(scores, labels)

        # reg_loss = 0
        # for param in self.model.parameters():
        #     reg_loss += self.l1_crit(param)

        # loss = loss + self.factor * reg_loss

        pred = scores.max(1)[1]
        num_correct = pred.data.eq(labels.data).sum()

        tp = pred.data.eq(labels.data).masked_select(labels.ne(0).data).sum()
        tn = pred.data.eq(labels.data).masked_select(labels.ne(1).data).sum()

        all_p = labels.data.eq(1).sum()
        all_n = labels.data.eq(0).sum()

        fn = all_p - tp
        fp = all_n - tn


        # accuracy: (TP+TN)/(TP+FN+FP+TN)
        # accuracy = num_correct * 1.0 / batch_size
        # precision: TP/(TP+FP)
        # if all_p == 0:
            # precision = 0.0
        # else:
            # precision = tp * 1.0 / all_p
        # recall: TP/(TP+FN)
        # recall = tp * 1.0 / (tp + fn)

        loss_data = loss.data[0]
        if not eval:
            loss.div(batch_size).backward()

        return loss_data, grad_output, num_correct, tp, tn, fn, fp
