from __future__ import division

import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time

import Model
from Dataset import Dataset
from Optim import Optim
from Criterion import Criterion

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=512,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=512,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")


# CNN parameters
## Encoder or Decoder
parser.add_argument("-hidden_size", type=int, default=512,
                    help="CNN hidden size")
parser.add_argument("-kernel_size", type=int, default=5,
                    help="")
parser.add_argument("-enc_layers", type=int, default=2,
                    help="Numbers of encoder hidden layer")

# Decoder
parser.add_argument("-dec_layers", type=int, default=1,
                    help="Numbers of decoder hidden layer")


## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

#learning rate
parser.add_argument('-learning_rate', type=float, default=0.01,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

#pretrained word vectors

parser.add_argument('-pre_word_vecs',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
# parser.add_argument('-pre_word_vecs_dec',
#                     help="""If a valid path is specified, then this will load
#                     pretrained word embeddings on the decoder side.
#                     See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

def eval(model, criterion, data, total_example_nums):
    total_loss = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        batch = data[i]
        labels = batch[2]
        scores = model(batch[0], batch[1])
        loss, _, num_correct, accuracy, precision, recall = criterion.loss(
                scores, labels, model.generator, eval=True)
        total_loss += loss
        total_num_correct += num_correct

    model.train()
    return total_loss, total_num_correct / total_example_nums, accuracy, precision, recall


def trainModel(model, trainData, validData, dataset, optim, criterion):
    print(model)
    model.train()

    total_example_nums = len(dataset['train']['question'])
    # define criterion of each GPU

    start_time = time.time()
    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct, report_num_example = 0, 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]
            model.zero_grad()

            labels = batch[2]
            scores = model(batch[0], batch[1])
            loss, gradOutput, num_correct, accuracy, precision, recall = criterion.loss(
                    scores, labels, model.generator)

            # scores.backward(gradOutput)

            # update the parameters
            optim.step()

            report_loss += loss
            report_num_correct += num_correct
            report_num_example += batch[1].size(1)
            total_loss += loss
            total_num_correct += num_correct

            # accuracy: (TP+TN)/(TP+FN+FP+TN)

            # precision

            # recall: TP/(TP+FN)


            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; loss: %6.2f; acc: %6.2f; %6.0f s elapsed; accuracy: %6.2f; precision: %6.2f; recall: %6.2f" %
                      (epoch, i+1, len(trainData),
                      loss,
                      report_num_correct / report_num_example * 100,
                      time.time()-start_time),
                      accuracy,
                      precision,
                      recall)

                report_loss = report_num_correct = report_num_example = 0
                start = time.time()

        return total_loss, total_num_correct / total_example_nums, accuracy, precision, recall

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc, train_accuracy, train_precision, train_recall = trainEpoch(epoch)
        print('Train acc: %g' % (train_acc*100))
        print('Train accuracy: %g' % (train_accuracy*100))
        print('Train precision: %g' % (train_precision*100))
        print('Train recall: %g' % (train_recall*100))


        #  (2) evaluate on the validation set
        valid_loss, valid_acc, valid_accuracy, valid_precision, valid_recall = eval(model, criterion, validData, len(dataset['test']['question']))
        print('Validation acc: %g' % (valid_acc*100))
        print('Validation accuracy: %g' % (valid_accuracy*100))
        print('Validation precision: %g' % (valid_precision*100))
        print('Validation recall: %g' % (valid_recall*100))


        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['word2index'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_prec_%.2f_rec_%.2f_e%d.pt' % (opt.save_model, 100*valid_accuracy, 100*valid_precision, 100*valid_recall, epoch))

def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['word2index'] = checkpoint['word2index']

    trainData = Dataset(dataset['train']['question'], dataset['train']['answer'],
                             dataset['train']['label'], opt.batch_size, opt.gpus)
    validData = Dataset(dataset['test']['question'], dataset['test']['answer'],
                             dataset['test']['label'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['word2index']
    print(' * vocabulary size: %d' % (len(dicts)))
    print(' * number of training sentences. %d' % len(dataset['train']['question']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    encoder = Model.Encoder(opt)
    decoder = Model.Decoder(opt)

    generator = nn.Sequential(
        nn.Linear(opt.dec_layers, 2),
        nn.LogSoftmax())

    model = Model.AnswerSelectModel(encoder, decoder, opt, len(dicts))

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s' % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        # encoder.load_pretrained_vectors(opt)
        # decoder.load_pretrained_vectors(opt)
        model.load_pretrained_vectors(opt)

        optim = Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    criterion = Criterion(model, opt)

    trainModel(model, trainData, validData, dataset, optim, criterion)


if __name__ == "__main__":
    main()