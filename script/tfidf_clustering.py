#!/usr/bin/env python
import sys
sys.path.append('..')

from bag_of_words import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import numpy as np

topic_num = 10
df = get_data(path='../data/iphone6.csv')

#define vectorizer parameters
f = open('stop_words.txt', 'r')
mystop = f.readlines()[0].split()
f.close()
mystop.extend( list(ENGLISH_STOP_WORDS) )

tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                   max_features=200000,
                                   min_df=1,
                                   use_idf=True,
                                   ngram_range=(1,1),
                                   stop_words = mystop,
                                   norm='l2')
tfidf_matrix = tfidf_vectorizer.fit_transform(df.Reviews_bw)
vocab = tfidf_vectorizer.get_feature_names()

cluster_method = KMeans(n_clusters=topic_num)
#cluster_method = SpectralClustering(n_clusters=topic_num)
cluster_method.fit(tfidf_matrix)

topics = cluster_method.labels_.tolist()
df['topic'] = topics
#print df['topics'].value_counts()

grouped = df['Rating'].groupby(df['topic'])
print grouped.mean()

print "Top terms per cluster:"
#sort cluster centers by proximity to centroid
order_centroids = cluster_method.cluster_centers_.argsort()[:, ::-1]
review_group = df['Reviews'].groupby(df['topic'])

for i in range(topic_num):
    print "\nCluster %d words:" % i
    for ind in order_centroids[i, :5]: #replace 5 with n words per cluster
        print vocab[ind],
    distances = cluster_method.transform(tfidf_matrix)[:, i]
    review_indices = np.argsort(distances)[::][:5] #get the closest 5 reviews
    for rindex in review_indices:
        print df.iloc[rindex]['Reviews']
