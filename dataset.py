from __future__ import division
import torch
import math
import random

import torch
from torch.autograd import Variable


class Dataset(object):

    def __init__(self, questions, answers, labels, batch_size, cuda, volatile=False):

        self.questions = questions
        self.answers = answers
        self.labels = labels
        assert (len(self.questions) == len(self.answers)) and (len(self.answers) == len(self.labels))
        self.batch_size = batch_size
        self.numBatches = math.ceil(len(self.questions)/batch_size)
        self.volatile = volatile
        self.cuda = cuda

    def _batchify(self, data, align_right=False, include_lengths=False, PADDING_TOKEN=0):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(PADDING_TOKEN)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        questions, lengths = self._batchify(
            self.questions[index*self.batch_size:(index+1)*self.batch_size])
        answers = self._batchify(
            self.answers[index*self.batch_size:(index+1)*self.batch_size])
        labels = self._batchify(
            self.labels[index*self.batch_size:(index+1)*self.batch_size])

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(questions))
        batch = zip(indices, questions, answers, labels)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, questions, answers, labels = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return wrap(questions), wrap(answers), wrap(labels)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.questions, self.answers, self.labels))
        self.questions, self.answers, self.labels = zip(*[data[i] for i in torch.randperm(len(data))])