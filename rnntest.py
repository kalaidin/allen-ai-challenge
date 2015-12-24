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
N_BATCH = 10


OUTPUT_DIM = 20
N_HIDDEN = 10
LEARNING_RATE = .001
GRAD_CLIP = 100
EPOCH_SIZE = 100
NUM_EPOCHS = 10


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
    rec = np.concatenate(vecs, axis=0).astype('float32')
    if rec.shape[0] > seq_length:
        rec = rec[rec.shape[0] - seq_length:, :]
    else:
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
# valid_i, valid_q, valid_c, valid_w = list(zip(*list(generate_batches([all_data[i] for i in valid_idx]))))
# valid_i = np.array(valid_i, dtype='int')
# valid_q = np.array(valid_q, dtype='float32')
# valid_c = np.array(valid_c, dtype='float32')
# valid_w = np.array(valid_w, dtype='float32')


print("Building question network ...")
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, QUESTION_LEN, W2V_DIM))
# l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, QUESTION_LEN))
l_forward = lasagne.layers.RecurrentLayer(
    l_in, N_HIDDEN,
    # mask_input=l_mask,
    grad_clipping=GRAD_CLIP,
    W_in_to_hid=lasagne.init.HeUniform(),
    W_hid_to_hid=lasagne.init.HeUniform(),
    nonlinearity=lasagne.nonlinearities.rectify, only_return_final=True)
l_out = lasagne.layers.DenseLayer(l_forward, num_units=OUTPUT_DIM, nonlinearity=lasagne.nonlinearities.rectify)

target_values = T.vector('target_output')

network_output = lasagne.layers.get_output(l_out)
predicted_values = network_output.flatten()
# Our cost will be mean-squared error
cost = T.mean((predicted_values - target_values)**2)
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out)
# Compute SGD updates for training
print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
# Theano functions for training and computing cost
print("Compiling functions ...")
train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                        cost, updates=updates)
compute_cost = theano.function(
    [l_in.input_var, target_values, l_mask.input_var], cost)



def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``

    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (n_batch, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (n_batch,).
    mask : np.ndarray
        A binary matrix of shape (n_batch, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.

    References
    ----------
    .. [1] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.

    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    mask = np.zeros((n_batch, max_length))
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype(theano.config.floatX))


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
    # The network also needs a way to provide a mask for each sequence.  We'll
    # use a separate input layer for that.  Since the mask only determines
    # which indices are part of the sequence for each batch entry, they are
    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    # Setting only_return_final=True makes the layers only return their output
    # for the final time step, which is all we need for this task
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True, backwards=True)
    # Now, we'll concatenate the outputs to combine them.
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    target_values = T.vector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    # The network output will have shape (n_batch, 1); let's flatten to get a
    # 1-dimensional vector of predicted values
    predicted_values = network_output.flatten()
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data()

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                X, y, m = gen_data()
                train(X, y, m)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()