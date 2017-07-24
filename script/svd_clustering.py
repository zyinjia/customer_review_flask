#!/usr/bin/env python
import sys
sys.path.append('..')

from bag_of_words import *
from topic_model_svd import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import numpy as np

topic_num = 10
df = get_data(path='../data/iphone6.csv')

f = open('stop_words.txt', 'r')
mystop = f.readlines()[0].split()
f.close()
mystop.extend( list(ENGLISH_STOP_WORDS) )

vectorizer = CountVectorizer(max_df=1.0,
                             max_features=200000,
                             min_df=1,
                             stop_words = mystop,
                             ngram_range=(1,1))
bw_matrix = vectorizer.fit_transform(df.Reviews_bw)
bw_matrix = TfidfTransformer(norm='l2', use_idf=False).fit_transform(bw_matrix)
vocab = vectorizer.get_feature_names()

t, s, d = svds(bw_matrix, k=50)

#print bw_matrix.shape
#print d.shape
#print t.shape
km = KMeans(n_clusters=topic_num)
km.fit(t)
topics = km.labels_.tolist()
df['topic'] = topics
#print df['topics'].value_counts()

grouped = df['Rating'].groupby(df['topic'])
print grouped.mean()

print "Top terms per cluster:"
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
review_group = df['Reviews'].groupby(df['topic'])

for i in range(topic_num):
    print "\nCluster %d words:" % i
    for ind in order_centroids[i, :2]: #replace 5 with n words per cluster
        print ind
        base = get_new_base(d[ind], vocab, cutoff=0.3)
        print [item[1] for item in base]
    distances = km.transform(t)[:, i]
    review_indices = np.argsort(distances)[::][:5] #get the closest 5 reviews
    for rindex in review_indices:
        print df.iloc[rindex]['Reviews']

#print topics
