#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import  print_function, division, unicode_literals

import codecs
import re
# from spacy.en import English
# nlp = English()

BIOLOGY_CONCEPTS = '/home/marat/data/allen-ai-challenge/biology_concepts.txt'

NEWPAGE = '\x0c'
GARBAGE = '\xa9'
TO_REMOVE = 'This content is available for free at'

START_LINE = '1 | INTRODUCTION TO BIOLOGY'


def first_filter(stream):
    started = False
    for line in f:
        if not started:
            if START_LINE == line.strip():
                started = True
            else:
                continue
        if (not line.strip()) or (line[0] == GARBAGE) or line.startswith(TO_REMOVE):
            continue
        yield line.strip()

page_breaks = re.compile(r'[\d]+ CHAPTER [\d]+ \| *')
re.match(page_breaks, '10 CH1APTER 1 | INTRODUCTION TO BIOLOGY')


def glue_pages(stream):
    older, old = '', next(stream)
    for new in stream:
        # print(repr(new))
        if re.match(page_breaks, new) == None:
            # print('-'*80, older, old, new)
            yield older + new
            older, old = next(stream), next(stream)
        else:
            yield older.strip()
            older = old
            old = new


with codecs.open(BIOLOGY_CONCEPTS, encoding='iso-8859-1') as f:
    i = 0
    for line in glue_pages(first_filter(f)):
        print(line)
        i += 1
        if i >= 20:
            break


    # i = 0
    # while 'Michael Rutledge Middle Tennessee State University' != f.readline():
    #     i += 1
    #     if i > 1000:
    #         break
    #
    # for _ in xrange(70):
    #     print(f.readline())