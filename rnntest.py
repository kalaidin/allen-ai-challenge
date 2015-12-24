#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import  print_function, division

import numpy as np
import theano
import theano.tensor as T
import lasagne
import re
from sklearn.cross_validation import KFold

TRAIN_FILE = '/home/marat/Downloads/training_set.tsv'
VALID_FILE = '/home/marat/Downloads/validation_set.tsv'
W2V_DICT_FILE = '/home/marat/Downloads/w2v_a2.tsv'

W2V_DIM = 300
QUESTION_LEN = 50
ANSWER_MAX_LEN = 20
BATCH_SIZE = 15


OUTPUT_DIM = 20
N_HIDDEN = 100
LEARNING_RATE = .001
GRAD_CLIP = 100
EPOCH_SIZE = 100
NUM_EPOCHS = 10
MARGIN = 0


def read_dict():
    d = {}
    with open(W2V_DICT_FILE) as f:
        for row in (line.strip().split('\t') for line in f):
            w = row[0]
            v = np.array(row[1:], dtype='float32')
            assert v.shape == (300,)
            d[w] = v
    return d

W2V = read_dict()


def text2words(t):
    s = re.sub(r'[^\w\s]', '', t)
    r = s.lower().split()
    return r

# TODO: generate mask vector
def text2vec(text, seq_length):
    vecs = []
    for w in text2words(text):
        try:
            vecs.append(W2V[w][np.newaxis, :])
        except KeyError:
            continue
    rec = np.concatenate(vecs, axis=0).astype('float32')
    if rec.shape[0] > seq_length:
        # trim long sentences
        rec = rec[rec.shape[0] - seq_length:, :]
    elif rec.shape[0] < seq_length:
        # extend short sentences with zeros
        rec = np.vstack([np.zeros((seq_length - rec.shape[0], rec.shape[1])), rec])
    assert rec.shape[0] == seq_length
    return rec


def read_data(file_name):
    data = []
    with open(file_name) as f:
        header = f.readline().strip().split('\t')
        print(header)
        for row in (line.strip().split('\t') for line in f):
            i, q, answer, aA, aB, aC, aD = row
            # if len(text2words(q))>140:
            #     print(q)
            # for a in [aA, aB, aC, aD]:
            #     if len(text2words(a)) > 20:
            #         print(a)
            try:
                text2vec(aA, ANSWER_MAX_LEN)
                text2vec(aB, ANSWER_MAX_LEN)
                text2vec(aC, ANSWER_MAX_LEN)
                text2vec(aD, ANSWER_MAX_LEN)
            except Exception as ex:
                print(ex)
                # print(row)
                continue
            data.append((i,
                         text2vec(q, QUESTION_LEN),
                         answer,
                         text2vec(aA, ANSWER_MAX_LEN),
                         text2vec(aB, ANSWER_MAX_LEN),
                         text2vec(aC, ANSWER_MAX_LEN),
                         text2vec(aD, ANSWER_MAX_LEN)))
    return data

all_data = read_data(TRAIN_FILE)
train_idx, valid_idx = KFold(len(all_data)).__iter__().next()


def generate_batches(data):
    for i, q, a, A, B, C, D in data:
        correct_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[a]
        correct_answer = [A, B, C, D][correct_index]
        wrong_answers = [A, B, C, D]
        del wrong_answers[correct_index]
        for wa in wrong_answers:
            yield i, q, correct_answer, wa

train_i, train_q, train_c, train_w = list(zip(*list(generate_batches([all_data[i] for i in train_idx]))))
train_i = np.array(train_i, dtype='int')
train_q = np.array(train_q, dtype='float32')
train_c = np.array(train_c, dtype='float32')
train_w = np.array(train_w, dtype='float32')

valid_i, valid_q, valid_c, valid_w = list(zip(*list(generate_batches([all_data[i] for i in valid_idx]))))
valid_i = np.array(valid_i, dtype='int')
valid_q = np.array(valid_q, dtype='float32')
valid_c = np.array(valid_c, dtype='float32')
valid_w = np.array(valid_w, dtype='float32')


print('Building question network ...')
q_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, QUESTION_LEN, W2V_DIM))
q_forward = lasagne.layers.RecurrentLayer(
    q_in, N_HIDDEN,
    # mask_input=l_mask,
    grad_clipping=GRAD_CLIP,
    W_in_to_hid=lasagne.init.HeUniform(),
    W_hid_to_hid=lasagne.init.HeUniform(),
    nonlinearity=lasagne.nonlinearities.rectify, only_return_final=True)
q_out = lasagne.layers.DenseLayer(q_forward, num_units=OUTPUT_DIM, nonlinearity=lasagne.nonlinearities.rectify)
target_values = T.vector('target_output')
question_point = lasagne.layers.get_output(q_out)
q_params = lasagne.layers.get_all_params(q_out)


print('Building siamese answer network ...')
sa_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, QUESTION_LEN, W2V_DIM))
sa_forward = lasagne.layers.RecurrentLayer(
    sa_in, N_HIDDEN,
    # mask_input=l_mask,
    grad_clipping=GRAD_CLIP,
    W_in_to_hid=lasagne.init.HeUniform(),
    W_hid_to_hid=lasagne.init.HeUniform(),
    nonlinearity=lasagne.nonlinearities.rectify, only_return_final=True)
sa_out = lasagne.layers.DenseLayer(sa_forward, num_units=OUTPUT_DIM, nonlinearity=lasagne.nonlinearities.rectify)
answer_point = lasagne.layers.get_output(sa_out)
sa_params = lasagne.layers.get_all_params(q_out)

correct_point = answer_point[0:BATCH_SIZE]
wrong_point = answer_point[BATCH_SIZE:]

print('Computing cost ...')

correct_dist = 1 - T.dot(question_point.T, correct_point) / (question_point.norm(2) * correct_point.norm(2))
wrong_dist = 1 - T.dot(question_point.T, wrong_point) / (question_point.norm(2) * wrong_point.norm(2))

cost = T.maximum(0, correct_dist - wrong_dist + 1).mean()

print('Computing updates ...')
updates = lasagne.updates.adam(cost, q_params)
updates.update(lasagne.updates.adam(cost, sa_params))

print('Compiling functions ...')
train_fn = theano.function([q_in.input_var, sa_in.input_var], cost, updates=updates)
cost_fn = theano.function([q_in.input_var, sa_in.input_var], cost)

print('Model compiled!')


indexi_train = np.arange(train_q.shape[0])
indexi_valid = np.arange(valid_q.shape[0])
for e in xrange(100):
    np.random.shuffle(indexi_train)
    train_costs = []
    for i in xrange(0, indexi_train.shape[0], BATCH_SIZE):
        keys = indexi_train[i:i + BATCH_SIZE]
        if keys.shape[0] != BATCH_SIZE:
            continue
        cost = train_fn(train_q[keys], np.vstack((train_c[keys], train_w[keys])))
        train_costs.append(cost)

    valid_costs = []
    for i in xrange(0, indexi_valid.shape[0], BATCH_SIZE):
        keys = indexi_valid[i:i + BATCH_SIZE]
        if keys.shape[0] != BATCH_SIZE:
            continue
        cost = cost_fn(valid_q[keys], np.vstack((valid_c[keys], valid_w[keys])))
        valid_costs.append(cost)

    print(e, np.mean(train_costs), np.mean(valid_costs))