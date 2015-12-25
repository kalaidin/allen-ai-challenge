#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import  print_function, division

from time import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import re
from sklearn.cross_validation import KFold

TRAIN_FILE = '/home/marat/Downloads/training_set.tsv'
TEST_FILE = '/home/marat/Downloads/validation_set.tsv'
W2V_DICT_FILE = '/home/marat/Downloads/w2v_a2.tsv'

SUBMISSION_FILE = '/home/marat/allen-ai-submission.csv'

W2V_DIM = 300
QUESTION_LEN = 50
ANSWER_MAX_LEN = 20
BATCH_SIZE = 15

OUTPUT_DIM = 100
N_HIDDEN = 100
GRAD_CLIP = 10
NUM_EPOCHS = 100
MARGIN = 0.1

# TODO: add L2
# TODO: add dropout
# TODO: add mask


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


def text2vec(text, seq_length):
    vecs = []
    for w in text2words(text):
        try:
            vecs.append(W2V[w][np.newaxis, :])
        except KeyError:
            continue
    if not vecs:
        vecs.append(np.random.normal(scale=1e-4, size=(1, W2V_DIM)).astype(dtype='float32'))
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
        print('Train file header:', ', '.join(header))
        for row in (line.strip().split('\t') for line in f):
            i, q, answer, aA, aB, aC, aD = row
            data.append((int(i),
                         text2vec(q, QUESTION_LEN),
                         answer,
                         text2vec(aA, ANSWER_MAX_LEN),
                         text2vec(aB, ANSWER_MAX_LEN),
                         text2vec(aC, ANSWER_MAX_LEN),
                         text2vec(aD, ANSWER_MAX_LEN)))
    return data


def read_submission_sample(file_name):
    with open(file_name) as f:
        header = f.readline().strip().split('\t')
        print('Submission header:', ', '.join(header))
        for row in (line.strip().split('\t') for line in f):
            i, q, aA, aB, aC, aD = row
            yield (int(i),
                   text2vec(q, QUESTION_LEN),
                   text2vec(aA, ANSWER_MAX_LEN),
                   text2vec(aB, ANSWER_MAX_LEN),
                   text2vec(aC, ANSWER_MAX_LEN),
                   text2vec(aD, ANSWER_MAX_LEN))


def generate_batches(data):
    for i, q, a, A, B, C, D in data:
        correct_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[a]
        correct_answer = [A, B, C, D][correct_index]
        wrong_answers = [A, B, C, D]
        del wrong_answers[correct_index]
        for wa in wrong_answers:
            yield i, q, correct_answer, wa


def zip_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape
    n = arr1.shape[0]
    z = np.zeros((n*2,) + arr1.shape[1:], dtype=arr1.dtype)
    for i in xrange(arr1.shape[0]):
        z[i*2] = arr1[i]
        z[i*2 + 1] = arr2[i]
    return z

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

correct_dist = 1 - T.sum(question_nvec * correct_nvec, axis=1)
wrong_dist = 1 - T.sum(question_nvec * wrong_nvec, axis=1)

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


with open(SUBMISSION_FILE, 'w') as f:
    f.write('id,correctAnswer\n')
    for i, q, A, B, C, D in read_submission_sample(TEST_FILE):
        q = np.repeat(q[np.newaxis], 2, axis=0)
        a = np.vstack([A[np.newaxis], B[np.newaxis], C[np.newaxis], D[np.newaxis]])
        d = dist_fn(q, a)
        d_A, d_B = d[0]
        d_C, d_D = d[1]
        answer_index = np.argmin([d_A, d_B, d_C, d_D])
        answer = 'ABCD'[answer_index]
        text = '%s,%s\n' % (i, answer)
        f.write(text)
        print(text[:-1], [d_A, d_B, d_C, d_D])
