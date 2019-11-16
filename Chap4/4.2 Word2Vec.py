%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


N = 5 # 윈도우 갯수
VEC_DIM = 100 # 임베딩 차원


txt_path = '../data/tokenized/korquad_mecab.txt'
corpus = [sent.strip().split() for sent in open(txt_path, 'r', encoding='utf-8').readlines()]

vocab = set()
for i in range(len(corpus)):
    vocab.update(corpus[i])

print('단어 갯수 : ' + str(len(vocab)))

word_to_id = dict()
for word in vocab:
    word_to_id[word] = len(word_to_id)

id_to_word = []
for word in word_to_id.keys():
    id_to_word.append(word)


def sent_to_ngram(n, sent):
    ngrams = []

    for i in range(n, len(sent) - n):
        ngram = []
        for j in range(-n, n + 1):
            ngram.append(sent[i + j])
        ngrams.append(ngram)

    return ngrams

train_data = []
for i in range(len(corpus)):
    train_data += sent_to_ngram(N, corpus[i])

def sent_to_vec(sent):
    vec = []
    for word in sent:
        vec.append(word_to_id[word])
    return vec


train_input = []
train_target = []
for data in train_data[0:2]:
    train_input.append(sent_to_vec(data[:N])+sent_to_vec(data[N+1:]))
    train_target.append(word_to_id[data[N]])


train = torch.utils.data.TensorDataset(torch.tensor(train_input), torch.tensor(train_target))