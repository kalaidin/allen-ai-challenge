#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import  print_function, division

from itertools import chain
from time import time

import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import l2, regularize_network_params
from sklearn.cross_validation import KFold
from common import *

EPS = 0
# EPS = 1e-5

SUBMISSION_FILE = '/home/marat/allen-ai-cnn_mean.csv'

QUESTION_LEN = 50
ANSWER_MAX_LEN = 20
BATCH_SIZE = 50

N_HIDDEN = 100
OUTPUT_NONLINEARITY = lasagne.nonlinearities.sigmoid

L2 = 1e-4
NUM_EPOCHS = 100
MARGIN = 1

# TODO: add dropout
# TODO: add mask

all_data = read_data(TRAIN_FILE)
kfold_iter = KFold(len(all_data)).__iter__()
kfold_iter.next()
kfold_iter.next()
train_idx, valid_idx = kfold_iter.next()

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
q_in = lasagne.layers.InputLayer(shape=(None, QUESTION_LEN, W2V_DIM))
q_n = lasagne.layers.DimshuffleLayer(q_in, (0, 2, 1))
q_n = lasagne.layers.Conv1DLayer(q_n, N_HIDDEN, filter_size=1, pad=0)
# q_n = lasagne.layers.dropout(q_n)
q_n = lasagne.layers.Conv1DLayer(q_n, 1, filter_size=7, pad=3, nonlinearity=OUTPUT_NONLINEARITY)
q_out = lasagne.layers.reshape(q_n, shape=([0], -1, 1))
question_vec = T.sum(lasagne.layers.get_output(q_out) * q_in.input_var, axis=1)
# question_vec = question_vec / (question_vec.norm(2, axis=1).dimshuffle((0, 'x')) + EPS)
# question_vec.eval({q_in.input_var: train_q[:13]}).shape
q_params = lasagne.layers.get_all_params(q_out)

lasagne.layers.get_output(q_out).eval({q_in.input_var: train_q[0:1]})

print('Building siamese answer network ...')
a_in = lasagne.layers.InputLayer(shape=(None, ANSWER_MAX_LEN, W2V_DIM))
a_n = lasagne.layers.DimshuffleLayer(a_in, (0, 2, 1))
a_n = lasagne.layers.Conv1DLayer(a_n, N_HIDDEN, filter_size=1, pad=0)
# a_n = lasagne.layers.dropout(a_n)
a_n = lasagne.layers.Conv1DLayer(a_n, 1, filter_size=7, pad=3, nonlinearity=OUTPUT_NONLINEARITY)
a_out = lasagne.layers.reshape(a_n, shape=([0], -1, 1))
answer_vec = T.sum(lasagne.layers.get_output(a_out) * a_in.input_var, axis=1)
# answer_vec = answer_vec / (answer_vec.norm(2, axis=1).dimshuffle((0, 'x')) + EPS)
a_params = lasagne.layers.get_all_params(a_out)
# lasagne.layers.get_output(a_out).eval({a_in.input_var: train_w[:1]})
# answer_vec.eval({a_in.input_var: train_c[:1]}).shape

correct_vec = answer_vec[0::2]
wrong_vec = answer_vec[1::2]

print('Computing cost ...')

# correct_cos_sim = T.sum(question_vec * correct_vec, axis=1)
# wrong_cos_sim = T.sum(question_vec * wrong_vec, axis=1)
correct_cos_sim = (question_vec-correct_vec).norm(2, axis=1)
wrong_cos_sim = (question_vec-wrong_vec).norm(2, axis=1)

cost_l2 = L2*regularize_network_params(a_out, l2) + L2*regularize_network_params(q_out, l2)
cost = T.maximum(0, (MARGIN + correct_cos_sim - wrong_cos_sim)).mean() + cost_l2
# cost = T.maximum(0, (MARGIN - T.sqr(correct_cos_sim) + T.sqr(wrong_cos_sim))).mean() + cost_l2

print('Computing updates ...')
updates = lasagne.updates.adam(cost, q_params)
updates.update(lasagne.updates.adam(cost, a_params))

print('Compiling functions ...')
train_fn = theano.function([q_in.input_var, a_in.input_var], cost, updates=updates)
cost_fn = theano.function([q_in.input_var, a_in.input_var], cost)
cost_l2_fn = theano.function([], cost_l2)
dist_fn = theano.function([q_in.input_var, a_in.input_var], [correct_cos_sim, wrong_cos_sim])
print('Model compiled!')


# def norm(ndarr, axis=2):
#     return ndarr / np.linalg.norm(ndarr, axis=axis, keepdims=True)

print('Training ...')
indexi_train = np.arange(train_q.shape[0])
indexi_valid = np.arange(valid_q.shape[0])
for e in xrange(NUM_EPOCHS):
    epoch_start = time()
    np.random.shuffle(indexi_train)

    valid_costs = []
    valid_correct_sum = 0
    valid_count = 0
    for i in xrange(0, indexi_valid.shape[0], BATCH_SIZE):
        keys = indexi_valid[i:i + BATCH_SIZE]
        valid_answers = zip_arrays(valid_c[keys], valid_w[keys])
        cost = cost_fn(valid_q[keys], valid_answers)
        d = dist_fn(valid_q[keys], valid_answers)
        valid_costs.extend([cost] * keys.shape[0])
        valid_correct_sum += (d[0] < d[1]).sum()
        valid_count += d[0].shape[0]
    valid_accuracy = valid_correct_sum / valid_count * 100

    train_costs = []
    for i in xrange(0, indexi_train.shape[0], BATCH_SIZE):
        keys = indexi_train[i:i + BATCH_SIZE]
        cost = train_fn(train_q[keys], zip_arrays(train_c[keys], train_w[keys]))
        train_costs.append(cost)

    curr_l2 = cost_l2_fn()

    time_passed = time() - epoch_start
    print(e, np.mean(valid_costs), '%.2f%%' % valid_accuracy, np.mean(train_costs), curr_l2, '%.0fsec' % time_passed)


# Forming submission file SUBMISSION_FILE
with open(SUBMISSION_FILE, 'w') as f:
    f.write('id,correctAnswer\n')
    for i, q, A, B, C, D in read_submission_sample(TEST_FILE):
        q = np.repeat(q[np.newaxis], 2, axis=0)
        a = np.vstack([A[np.newaxis], B[np.newaxis], C[np.newaxis], D[np.newaxis]])
        d = list(chain(*dist_fn(q, a)))
        answer_index = np.argmin(d)
        answer = 'ABCD'[answer_index]
        text = '%s,%s\n' % (i, answer)
        f.write(text)
        print(text[:-1], d)


# (valid_q[:1] / (np.linalg.norm(valid_q[:1], axis=2, keepdims=True) + 1e-10)).sum(axis=2)
# np.linalg.norm(valid_q[:1], axis=2, keepdims=True)
