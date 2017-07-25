#!/usr/bin/env python
import sys
sys.path.append('..')

from bag_of_words import *
from topic_model_svd import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import numpy as np

topic_num = 10
df = get_data(path='../data/iphone6.csv')
#reviews_bw = df['Reviews_bw'].tolist()
#reviews_bw = [get_noun_text(text) for text in reviews_bw]

f = open('stop_words.txt', 'r')
mystop = f.readlines()[0].split()
f.close()
mystop.extend( list(ENGLISH_STOP_WORDS) )
#mystop = list(ENGLISH_STOP_WORDS)

vectorizer = CountVectorizer(max_df=1.0,
                             max_features=2000,
                             min_df=1,
                             stop_words = mystop,
                             ngram_range=(1,1))
bw_matrix = vectorizer.fit_transform(df.Reviews_bw)
bw_matrix = normalize(bw_matrix, norm='l2') #bw_matrix is a np sparse matrix
vocab = vectorizer.get_feature_names()

t, s, d = svds(bw_matrix, k=50) #d is ndarray

reviews = find_review_in_topics(d, bw_matrix, df, reviews_num=3, topic_num=topic_num, minwords=3)

for i in range(topic_num):
    print i
    print get_new_base(d[i], vocab, cutoff=0.2)
    print reviews[i]
