#!/usr/bin/env python
import sys
sys.path.append('..')

from bag_of_words import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

topic_num = 20
df = get_data(path='../data/iphone6.csv')
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                   max_features=200000,
                                   min_df=1,
                                   use_idf=True,
                                   ngram_range=(1,1),
                                   stop_words = 'english',
                                   norm='l2')
tfidf_matrix = tfidf_vectorizer.fit_transform(df.Reviews_bw)
vocab = tfidf_vectorizer.get_feature_names()

km = KMeans(n_clusters=topic_num)
km.fit(tfidf_matrix)

topics = km.labels_.tolist()
df['topics'] = topics
#print df['topics'].value_counts()

grouped = df['Rating'].groupby(df['topics'])
print grouped.mean()

print "Top terms per cluster:"
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
review_group = df['Reviews'].groupby(df['topics'])

for i in range(topic_num):
    print "\nCluster %d words:" % i
    for ind in order_centroids[i, :5]: #replace 6 with n words per cluster
        print vocab[ind],

    #print "Cluster %d review:" % i
    #for review in review_group.get_group(i):
    #    print '%s' % review
