#!/usr/bin/env python
import sys
sys.path.append('..')

from bag_of_words import *
from topic_model_svd import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return np.dot(Y, np.transpose(X))

# monkey patch (ensure cosine dist function is used)
k_means_.euclidean_distances = new_euclidean_distances
print 'a'


topic_num = 6
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
bw_matrix = normalize(bw_matrix, norm='l2')
vocab = vectorizer.get_feature_names()

t, s, d = svds(bw_matrix, k=20)

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
minwords = 2

for i in range(topic_num):
    print "\nCluster %d words:" % i
    for ind in order_centroids[i, :3]: #replace 5 with n words per cluster
        print ind, km.cluster_centers_[i, ind]
        base = get_new_base(d[ind], vocab, cutoff=0.2)
        print sorted(base, reverse=True)
    distances = km.transform(t)[:, i]
    review_sort = np.argsort(distances)[::]
    ##get the closest 5 reviews with length > minwords
    reviews = []
    r = 0
    while len(reviews)<5:
        index = review_sort[r]
        review = df.loc[index, 'Reviews']
        if len(review.split()) > minwords:
            reviews.append(review)
        r += 1
    print reviews

#print topics
