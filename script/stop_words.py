from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import nltk
import matplotlib.pyplot as plt
import numpy as np

def create_my_stopwords(vocab_clean, fstop, faccept):
    mystop = []
    myaccept = faccept.readlines()[0].split()
    for word in vocab_clean:
        if word in myaccept:
            continue
        print word
        if raw_input():
            mystop.append(word)
            fstop.write('\t'+word)
    return mystop

def filter_word(vocab):
    f = open('stop_words.txt', 'r')
    mystop = f.readlines()[0].split()
    exclude_tag = set(['RB', 'EX', 'IN', 'DT', 'CC', 'MD', 'RP'])
    tags = nltk.pos_tag(vocab)
    select_indices = []
    for i in range(len(vocab)):
        if (tags[i][1] not in exclude_tag) and (vocab[i] not in mystop):
            select_indices.append(i)
    return [vocab[i] for i in select_indices]

if __name__ == "__main__":
    from bag_of_words import *
    
    df = get_data()
    df_train, df_test, df_predict = split_df(df, 50000, 1000, 1000)
    
    # Get the full vocabulary
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 900)
    train_features = get_train_features_bw(df_train.Reviews_bw, vectorizer)
    vocab = vectorizer.get_feature_names()
    vocab_clean = filter_word(vocab)
    
    f  = open('stop_words.txt', 'a')
    fa = open('accept_words.txt', 'r')
    mystop = create_my_stopwords(vocab_clean, fstop=f, faccept=fa)    
    print mystop
    f.close()
    fa.close()
    
    
    
    