#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import codecs
import nltk
from nltk.tokenize import sent_tokenize

DATA = '/home/marat/ck12.txt'
DATA_CLEAN = '/home/marat/ck12_clean.txt'


def read_data(dirty_file):
    t = nltk.tokenize.TreebankWordTokenizer()

    state = 'intro'
    title = ''
    with codecs.open(dirty_file, encoding='utf8') as f:
        [f.readline() for _ in range(45)]
        for line in (line.strip() for line in f):
            if not line:
                continue
            words = t.tokenize(line)
            if len(words) < 5 and '.' not in line:
                if line == 'Summary':
                    state = 'summary'
                # yield '--S ' + line
                continue
            if len(words) == 2 and words[0] == 'Figure':
                state = 'figure'
                continue

            if state == 'figure':
                state = ''
                # yield '--F ' + line
                continue

            if len(words) < 15:
                state = 'title'
                title = line if line[-1] in '.?"' else line + '.'
                # yield '--T ' + line
                continue

            # data = title + ' ' + line if state == 'title' else line  # this seems to decrease w2v accuracy
            data = line
            state = 'data'

            sents = [s for s in sent_tokenize(data) if (s.find('http:') == -1) and (s.find('https:') == -1)]

            if len(sents) > 0:
                yield ' '.join(sents)


# for i, d in enumerate(read_data(DATA)):
#     print(d)

if __name__ == '__main__':
    with codecs.open(DATA_CLEAN, encoding='utf8', mode='w') as f:
        for i, d in enumerate(read_data(DATA)):
            print(d, file=f)

