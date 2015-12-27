#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import  print_function, division

from itertools import chain
from time import time

import theano
import theano.tensor as T
import lasagne
from sklearn.cross_validation import KFold
from common import *

SUBMISSION_FILE = '/home/marat/allen-ai-submission.csv'

QUESTION_LEN = 50
ANSWER_MAX_LEN = 20
BATCH_SIZE = 50

OUTPUT_DIM = 50
N_HIDDEN = 50

GRAD_CLIP = 10
NUM_EPOCHS = 100
MARGIN = np.exp(-1) + 0.1

# TODO: add L2
# TODO: add dropout
# TODO: add mask

all_data = read_data(TRAIN_FILE)
train_idx, valid_idx = KFold(len(all_data)).__iter__().next()

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
q_forward = lasagne.layers.RecurrentLayer(
    q_in, N_HIDDEN,
    # mask_input=l_mask,
    grad_clipping=GRAD_CLIP,
    W_in_to_hid=lasagne.init.HeUniform(),
    W_hid_to_hid=lasagne.init.HeUniform(),
    nonlinearity=lasagne.nonlinearities.rectify, only_return_final=True)
q_out = lasagne.layers.DenseLayer(q_forward, num_units=OUTPUT_DIM, nonlinearity=lasagne.nonlinearities.rectify)
target_values = T.vector('target_output')
question_vec = lasagne.layers.get_output(q_out)
question_nvec = question_vec / question_vec.norm(2, axis=1).dimshuffle(0, 'x')
q_params = lasagne.layers.get_all_params(q_out)


print('Building siamese answer network ...')
sa_in = lasagne.layers.InputLayer(shape=(None, QUESTION_LEN, W2V_DIM))
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

correct_vec = answer_point[0::2]
correct_nvec = correct_vec / correct_vec.norm(2, axis=1).dimshuffle((0, 'x'))
wrong_vec = answer_point[1::2]
wrong_nvec = wrong_vec / wrong_vec.norm(2, axis=1).dimshuffle((0, 'x'))

print('Computing cost ...')

correct_dist = T.exp(T.sum(question_nvec * correct_nvec, axis=1))
wrong_dist = T.exp(T.sum(question_nvec * wrong_nvec, axis=1))

cost = T.maximum(0, correct_dist - wrong_dist + MARGIN).mean()

print('Computing updates ...')
updates = lasagne.updates.adam(cost, q_params)
updates.update(lasagne.updates.adam(cost, sa_params))

print('Compiling functions ...')
train_fn = theano.function([q_in.input_var, sa_in.input_var], cost, updates=updates)
cost_fn = theano.function([q_in.input_var, sa_in.input_var], cost)
dist_fn = theano.function([q_in.input_var, sa_in.input_var], [correct_dist, wrong_dist])
print('Model compiled!')

print('Training ...')
indexi_train = np.arange(train_q.shape[0])
indexi_valid = np.arange(valid_q.shape[0])
for e in xrange(NUM_EPOCHS):
    epoch_start = time()
    np.random.shuffle(indexi_train)
    train_costs = []
    for i in xrange(0, indexi_train.shape[0], BATCH_SIZE):
        keys = indexi_train[i:i + BATCH_SIZE]
        cost = train_fn(train_q[keys], zip_arrays(train_c[keys], train_w[keys]))
        train_costs.append(cost)

    valid_costs = []
    for i in xrange(0, indexi_valid.shape[0], BATCH_SIZE):
        keys = indexi_valid[i:i + BATCH_SIZE]
        cost = cost_fn(valid_q[keys], zip_arrays(valid_c[keys], valid_w[keys]))
        valid_costs.append(cost)

    time_passed = time() - epoch_start
    print(e, np.mean(train_costs), np.mean(valid_costs), '%.0fsec' % time_passed)


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
