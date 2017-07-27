from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
import random


def filter_word(vocab):
    f = open('stop_words.txt', 'r')
    mystop = f.readlines()[0].split()
    select_indices = []
    for i in range(len(vocab)):
        if vocab[i] not in mystop:
            select_indices.append(i)
    return [vocab[i] for i in select_indices]


def vec_dir(vec):
    if sum(vec) > 0:
        return True #positive
    else:
        return False #negative


def find_review_in_topics(d, features_normalized, df_train, reviews_num=3, topic_num=20, minwords=3):
    '''
    reviews: a list of tuples, the tuple is product_name (string) + review (string)
    '''
    reviews = []
    for j in range(topic_num):
        topic_vec = d[j]
        select = get_relavant_reviews(topic_vec,
                                      features_normalized,
                                      df_train,
                                      reviews_num=reviews_num,
                                      minwords=minwords)
        reviews.append(select)
    return reviews

def get_relavant_reviews(topic_vec, features_normalized, df_train, reviews_num=3, minwords=3):
    if not vec_dir(topic_vec):
        topic_vec = -topic_vec
    sim_vec = csr_matrix(topic_vec).dot(np.transpose(features_normalized))
    sim_sort = sim_vec.toarray().reshape(-1).argsort()[::-1]
    select = []
    r = 0
    while len(select)<reviews_num:
        index = sim_sort[r]
        #print sim_vec.toarray().reshape(-1)[index] #similarity
        review = df_train.loc[df_train.index[index],'Reviews']
        if len(review.split()) > minwords:
            select.append(review)
        r += 1
    return select


def get_new_base(topic_vec, vocab, cutoff=0.2):
    '''
    return a list of tuples: each tuple contains the coefficient of the word, and the string of the word
    '''
    word_list = []
    if not vec_dir(topic_vec):
        topic_vec = -topic_vec
    for i in range(len(topic_vec)):
        if topic_vec[i]>cutoff:
            word_list.append( (topic_vec[i], vocab[i]) )
    return word_list


def get_top_words(topics):
    '''
    return a list of key words
    '''
    top_words = []
    for key in topics:
        for item in topics[key]:
            if item[1] not in top_words:
                top_words.append(item[1])
            else:
                continue
    return top_words

def make_fig_topics_trans(topics, eigenvalues, output):
    top_words = get_top_words(topics)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.grid(color='grey', linestyle='-', linewidth=0.5)

    topic_num = len(topics.keys())
    word_num  = len(top_words)

    X, Y, S, C = [], [], [], []
    for x in topics:
        for item in topics[x]:
            word = item[1]
            radi = item[0]
            X.append(x)
            Y.append(top_words.index(word))
            S.append(np.pi*abs(radi)**2*100)
            C.append(eigenvalues[x])

    ax.scatter(Y, X, s=S,
               c=C, cmap = plt.get_cmap('viridis'),
               alpha=0.5)

    ax.xaxis.tick_top()
    ax.set_ylim(-1, topic_num)
    ax.set_yticks(range(topic_num))
    ax.set_yticklabels(['topic%d' %i for i in range(topic_num)])

    ax.set_xlim(-1, word_num)
    ax.set_xticks(range(word_num))
    ax.set_xticklabels([top_words[i] for i in range(word_num)], rotation=40, ha='center')
    ax.invert_yaxis()
    fig.savefig( output )

def make_fig_topics(topics, eigenvalues, output=False):
    top_words = get_top_words(topics)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.grid(color='grey', linestyle='-', linewidth=0.5)

    topic_num = len(topics.keys())
    word_num  = len(top_words)

    X, Y, S, C = [], [], [], []
    for x in topics:
        for item in topics[x]:
            word = item[1]
            radi = item[0]
            X.append(x)
            Y.append(top_words.index(word))
            S.append(np.pi*abs(radi)**2*500)
            C.append(eigenvalues[x])

    ax.scatter(X, Y, s=S,
               c=C, cmap = plt.get_cmap('viridis'),
               alpha=0.5)

    ax.xaxis.tick_top()
    ax.set_xlim(-1, topic_num)
    ax.set_xticks(range(topic_num))
    ax.set_xticklabels(['Topic%d' %i for i in range(topic_num)], rotation=40, ha='center')

    ax.set_ylim(-1,word_num)
    ax.set_yticks(range(word_num))
    ax.set_yticklabels([top_words[i] for i in range(word_num)])
    ax.invert_yaxis()
    fig.tight_layout()
    if output:
        fig.savefig( output )
    return fig


def make_fig_reviews(topics, reviews, review_num, output):
    fig, axes = plt.subplots(nrows=int(math.ceil(review_num/5)), ncols=5,
                             figsize=(10, 5.5))
    for r in range(review_num):
        review = reviews[r][1]
        product = reviews[r][0]
        i = r/5
        j = r%5
        for item in topics[r]:
            word = item[1]
            radi = item[0]
            x, y = random.random(), random.random()
            axes[i, j].scatter(x, y, s=abs(radi)**2*2000,
                               c='grey', alpha=0.5)
            axes[i, j].text(x, y, word, horizontalalignment='center')
            axes[i, j].set_ylim(-0.8,1.8)
            axes[i, j].set_yticks([])
            axes[i, j].set_xticks([])
    fig.tight_layout()
    fig.savefig( output )


def get_reviews(path='./data/iphone6.csv'):
    from bag_of_words import get_data, get_train_features_bw
    df_train = get_data(path)
    topic_num = 10
    max_features = 2000

    # Get the full vocabulary
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = max_features+2000)
    train_features = get_train_features_bw(df_train.Reviews_bw, vectorizer)
    vocab = vectorizer.get_feature_names()

    vocab_clean = filter_word(vocab)
    vectorizer2 = CountVectorizer(analyzer = "word",   \
                                  tokenizer = None,    \
                                  preprocessor = None, \
                                  stop_words = None,   \
                                  vocabulary= vocab_clean,
                                  max_features = max_features)
    train_features = vectorizer2.transform(df_train.Reviews_bw)
    train_features = train_features.toarray()
    train_features_normalized = normalize(train_features, norm='l2', axis=1)
    t, s, d = np.linalg.svd(train_features_normalized, full_matrices=False)

    max_sim_list, max_ind_list, reviews = find_review_in_topics(d, train_features_normalized, df_train, num=topic_num)
    return reviews

if __name__ == "__main__":
    print get_reviews()

    '''
    topics = {}
    for j in range(topic_num):
        topics[j] = get_topic(d[j], vocab_clean)

    top_words = get_top_words(topics, vocab_clean)
    '''
    # plot
    #make_fig_topics(topics, top_words, s, 'topic_model_blu.svg')
    #make_fig_topics_trans(topics, top_words, s, 'topic_model_blu_trans.svg')

    '''
    f = open('topic_model_svd_review.txt', 'w')
    cnt = 0
    for review in reviews:
        f.write('%d\t%s\t%s' %(cnt, review[0], review[1]), encode='utf-8')
        cnt += 1
    f.close()
    '''
    #print reviews
    #make_fig_reviews(topics, reviews, review_num=15, output='./topic_model_review_blu.svg')
