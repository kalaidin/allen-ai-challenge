#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import  print_function, division

import numpy as np
import re

TRAIN_FILE = '/home/marat/Downloads/training_set.tsv'
TEST_FILE = '/home/marat/Downloads/validation_set.tsv'
W2V_DICT_FILE = '/home/marat/Downloads/w2v_a2.tsv'
W2V_DIM = 300

SUBMISSION_FILE = '/home/marat/allen-ai-submission.csv'
QUESTION_LEN = 50
ANSWER_MAX_LEN = 20


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
        r = np.random.uniform(-1e2, 1e2, size=(1, W2V_DIM))
        r = r / np.linalg.norm(r)
        vecs.append(r.astype(dtype='float32'))
        # vecs.append(np.zeros((1, W2V_DIM), dtype='float32'))
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


