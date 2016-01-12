#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals

from collections import Counter
from random import shuffle

from spacy.en import English
nlp = English()
import os
import codecs
from gensim.models.doc2vec import LabeledSentence, Doc2Vec, TaggedDocument, TaggedLineDocument, train_document_dm
from gensim.models.doc2vec import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = os.path.join(os.environ['HOME'], 'data', 'allen-ai-challenge')
CORPUS_RAW = os.path.join(DATA_DIR, 'ck12_clean.txt')
CORPUS = os.path.join(DATA_DIR, 'ck12_clean_tokenized.txt')
TRAINING_SET = os.path.join(DATA_DIR, 'training_set.tsv')


def tokenize(text):
    return [w.lemma_ for w in nlp(text) if w.is_alpha and not w.is_stop]

if not os.path.isfile(CORPUS):
    with codecs.open(CORPUS_RAW, encoding='utf8') as corpus_in:
        with codecs.open(CORPUS, mode='wb', encoding='utf8') as corpus_out:
            for line in corpus_in:
                if '↓' in line:
                    continue
                print(' '.join(tokenize(line)), file=corpus_out)
    print('Created corpus file "%s"' % CORPUS)


class ShufflingOrderDocs(object):
    def __init__(self, source):
        self.documents = []
        self.source = source
        with codecs.open(source, encoding='utf8') as f:
            for item_no, line in enumerate(f):
                self.documents.append(TaggedDocument(line.strip().split(), [item_no]))

    def __iter__(self):
        for doc in self.documents:
            yield doc
        shuffle(self.documents)


DOC2VEC_ITER_COUNT = 7
data = ShufflingOrderDocs(CORPUS)
model = Doc2Vec(sample=0.0001, negative=15)  # use fixed learning rate
model.build_vocab(data)
for epoch in range(DOC2VEC_ITER_COUNT):
    model.train(data)
    model.alpha *= 0.991  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


def infer_vector(words, model, starting_vector, iters=5, alpha=0.01):
    work = zeros(model.layer1_size, dtype=REAL)
    doctag_vectors = empty((1, model.vector_size), dtype=REAL)
    doctag_vectors[0] = np.array(starting_vector)
    doctag_locks = ones(1, dtype=REAL)

    neu1 = matutils.zeros_aligned(model.layer1_size, dtype=REAL)
    for i in range(iters):
        train_document_dm(model, a, [0], alpha, work, neu1,
                          learn_words=False, learn_hidden=False,
                          doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
    return doctag_vectors[0]

Q_TAGS_OBTAIN = 1
A_TAGS_OBTAIN = 100
DATA_TAGS_OBTAIN = 1
AGG_FUNCTION = np.mean
corrects = []
with codecs.open(TRAINING_SET, encoding='utf8') as f:
    f.readline()
    for lid, line in enumerate(f):
        qid, q, correct, A, B, C, D = line.strip().split('\t')

        q_ws = tokenize(q)
        doctags = [model.infer_vector(q_ws, steps=100, alpha=0.001) for _ in range(Q_TAGS_OBTAIN)]
        data_ids, data_sims = zip(*model.docvecs.most_similar(doctags, topn=DATA_TAGS_OBTAIN))
        datavecs = [model.docvecs[d] for d in data_ids]


        attempts = np.zeros((A_TAGS_OBTAIN, 4))
        for aid, answer_text in enumerate([A, B, C, D]):
            a = tokenize(answer_text)
            # ansvecs = [model[w] for w in a]
            # sum_answer = sum(ansvecs)
            # print(cosine_similarity(sum_answer, simvecs[0]))

            # scores = []
            # for qvec in datavecs:
            #     for _ in range(A_TAGS_OBTAIN):
            #         avec = model.infer_vector(tokenize(answer_text))
            #         scores.append(cosine_similarity(avec, qvec)[0, 0])
            # answer_scores.append(AGG_FUNCTION(scores))
            for att in range(A_TAGS_OBTAIN):
                avec = infer_vector(a, model, datavecs[0], iters=5, alpha=0.003)
                # avec = infer_vector(a, model, avec, iters=5, alpha=0.005)
                score = cosine_similarity(avec, datavecs[0])[0, 0]
                attempts[att, aid] = score

        c = Counter(attempts.argmax(axis=1))
        # for aid, a in enumerate([A, B, C, D]):
        #     print(c[aid], attempts.mean(axis=0)[aid], a)
        sum = attempts.mean(axis=0)
        rank = np.array([c[k] for k in range(4)])

        corrects.append(1 if 'ABCD'[np.argmax(rank*sum)] == correct else 0)

        print(lid, np.mean(corrects))
        break
        # if lid>=100:
        #     break




print(q)
id2sim = dict(zip(data_ids, data_sims))
for d in data.documents:
    if d.tags[0] in data_ids:
        print(id2sim[d.tags[0]], d)

print(model.most_similar('dwarf'))