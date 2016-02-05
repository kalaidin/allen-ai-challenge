# coding: utf-8
"""
created by artemkorkhov at 2016/01/19
"""

import os
import re
import json
import urllib
from itertools import chain

import wikipedia as wiki
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


ck12_url_topic = ['https://www.ck12.org/earth-science/', 'http://www.ck12.org/life-science/',
                  'http://www.ck12.org/physical-science/', 'http://www.ck12.org/biology/',
                  'http://www.ck12.org/chemistry/', 'http://www.ck12.org/physics/']

wiki_docs_dir = '/Users/artemkorkhov/Projects/kaggle/deephack/data/parsed_wiki_data/'


def get_wiki_docs():
    """
    """
    ensure_dir(wiki_docs_dir)
    topic_index = {}
    ck12_keywords = set()
    for url_topic in ck12_url_topic:
        topic = [p for p in url_topic.split('/') if p][-1]
        keywords= get_keyword_from_url_topic(url_topic)
        for kw in keywords:
            ck12_keywords.add(kw)
        topic_index.setdefault(topic, list(keywords))

    with open(os.path.join(wiki_docs_dir, "topic_index.json"), 'w+') as f:
        f.write(json.dumps(topic_index))
    #get and save wiki docs
    get_save_wiki_docs(ck12_keywords, wiki_docs_dir)


def get_save_wiki_docs(keywords, save_folder='data/parsed_wiki_data/'):
    """
    :param keywords:
    :param save_folder:
    :return:
    """
    doc_tree = {}
    fails = []
    ensure_dir(save_folder)
    n_total = len(keywords)
    print '# doc', 'total', '%', 'keywords', '# options'
    for i, kw in enumerate(keywords):
        suggested_docs = []
        if i <= 1244:
            print i
            continue
        kw = kw.lower()
        options = wiki.search(kw)
        suggested_docs.extend(options)
        print '=>', i, n_total, i * 1.0 / n_total, kw,
        for ix, page in enumerate(list(chain([kw], suggested_docs))):
            print ix, page, '(related to %s)' % kw
            try:
                content = wiki.page(page).content.encode('ascii', 'ignore')
            except wiki.exceptions.DisambiguationError as e:
                print 'DisambiguationError', kw, e
            except Exception as e:
                print "Error:", e
            if not content:
                continue
            page_name = page.replace('/', '_').lower().split()
            try:
                with open(os.path.join(save_folder, '_'.join(page_name) + '.txt'), 'w') as f:
                    f.write(content)
            except:
                fails.append(page_name)
                pass
        doc_tree.setdefault(kw, suggested_docs)
    with open(os.path.join(wiki_docs_dir, "keyword_related_index.json"), 'w+') as f:
        f.write(json.dumps(doc_tree))
    print 'FINISHED! \n Failed with: %s' % fails


def get_keyword_from_url_topic(url_topic):
    # Topic includes: Earth Science, Life Science, Physical Science, Biology, Chemestry and Physics
    lst_url = []
    html = urllib.urlopen(url_topic).read()
    soup = BeautifulSoup(html, 'html.parser')
    for tag_h3 in soup.find_all('h3'):
        url_res =  ' '.join(tag_h3.li.a.get('href').strip('/').split('/')[-1].split('-'))
        lst_url.append(url_res)
    return lst_url


def tokenize(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # 1. Remove non-letters
    review_text = re.sub(r"[^a-zA-Z]", " ", review)
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    # 3. Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    # 5. Return a list of words
    return words


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
