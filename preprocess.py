# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
import argparse
import os
import sys
import codecs
import torch

import jieba

parser = argparse.ArgumentParser(description='preprocess.py')
##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_file', type=str, default="./data/train.txt",
                    help="Path to the training source data")
parser.add_argument('-dev_file', type=str, default="./data/dev.txt",
                    help="Path to the training target data")

parser.add_argument('-save_data', type=str, default="./data/imdb",
                    help="Output file for the prepared data")

parser.add_argument('-maximum_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")

parser.add_argument('-vocab',
                    help="Path to an existing vocabulary")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=1000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(fileName):
    examples = [s.strip().split("	") for s in codecs.open(fileName, "r", encoding='utf-8-sig').readlines()]
    questions = [list(jieba.cut(clean_str(example[1]))) for example in examples]
    answers = [list(jieba.cut(clean_str(example[2]))) for example in examples]
    labels = [int(example[0]) for example in examples]
    return questions, answers, labels


def build_vocab(sequence, maximum_vocab_size=50000):
    word_count = Counter(itertools.chain(*sequence)).most_common(maximum_vocab_size)
    word2count = dict([(word[0], word[1]) for word in word_count])

    word2index = dict([(word, index+2) for index, word in enumerate(word2count)])
    word2index[0], word2index[1] = 0, 1

    index2word = dict([(index+2, word) for index, word in enumerate(word2count)])
    index2word[0], index2word[1] = 0, 1
    return word2count, word2index, index2word

def makeData(questions, answers, labels, word2index, shuffle=opt.shuffle):
    assert len(questions) == len(answers) and len(answers) == len(labels)
    sizes = []
    for idx in range(len(questions)):
        questions[idx] = torch.LongTensor([word2index[word] if word in word2index else 0 for word in questions[idx]])
        answers[idx] = torch.LongTensor([word2index[word] if word in word2index else 0 for word in answers[idx]])
        sizes += [len(questions)]
        labels[idx] = torch.LongTensor([labels[idx]])

    if shuffle == 1:
        print("... shuffling sentences")
        perm = torch.randperm(len(questions))
        questions = [questions[idx] for idx in perm]
        answers = [answers[idx] for idx in perm]
        labels = [labels[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print("... sorting sentences")
    _, perm = torch.sort(torch.Tensor(sizes))
    questions = [questions[idx] for idx in perm]
    answers = [answers[idx] for idx in perm]
    labels = [labels[idx] for idx in perm]
    return questions, answers, labels

def main():


    questions_train, answers_train, labels_train = load_data_and_labels(opt.train_file)
    questions_dev, answers_dev, labels_dev = load_data_and_labels(opt.dev_file)

    word2count, word2index, index2word = build_vocab(questions_train + answers_train + questions_dev + answers_dev, opt.maximum_vocab_size)

    print('Preparing training ...')
    train = {}
    train["question"], train["answer"], train["label"] = makeData(questions_train, answers_train, labels_train, word2index)

    print('Preparing validation ...')
    valid = {}
    valid['question'], valid['answer'], valid["label"]  = makeData(questions_dev, answers_dev, labels_dev, word2index)

    print("saving data to \'" + opt.save_data + ".train.pt\'...")
    save_data = {
        "train": train,
        "test": valid,
        "word2index": word2index
    }
    torch.save(save_data, opt.save_data + ".train.pt")

if __name__ == '__main__':
    main()