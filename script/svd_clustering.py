#!/usr/bin/env python
import sys
sys.path.append('..')

from bag_of_words import *
from topic_model_svd import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
#t, s, d = np.linalg.svd(bw_matrix.toarray(), full_matrices=False) #d is ndarray

#print bw_matrix.shape
#print d.shape
#print t.shape
km = KMeans(n_clusters=topic_num)
km.fit(t)
topics = km.labels_.tolist()
df['topic'] = topics
review_cnt = df['topic'].value_counts().tolist()

grouped = df['Rating'].groupby(df['topic'])
stars = grouped.mean().tolist()

print "Top terms per cluster:"
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
minwords = 2

select_reviews = []
topic_words = []
for i in range(topic_num):
    print "\nCluster %d words:" % i
    words = []
    for ind in order_centroids[i, :3]:
        print ind, km.cluster_centers_[i, ind]
        base = get_new_base(d[ind], vocab, cutoff=0.2)
        words.extend([item[1] for item in sorted(base, reverse=True)])
    print words
    topic_words.append(", ".join(words[:5]))
    distances = km.transform(t)[:, i]
    review_sort = np.argsort(distances)[::]
    ##get the closest 3 reviews with length > minwords
    reviews = []
    r = 0
    while len(reviews)<3:
        index = review_sort[r]
        review = df.loc[index, 'Reviews']
        if len(review.split()) > minwords:
            reviews.append(review)
            print review
        r += 1
    select_reviews.append("\n\n".join(reviews))

select_df = pd.DataFrame({
    'Topic': topic_words,
    'StarRating': ['%0.1f' %star for star in stars],
    'ReviewNum': review_cnt,
    'Reviews': select_reviews
    })

select_df.to_csv('../html_data/iphone6.csv')
