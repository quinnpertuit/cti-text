from model import *
from utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='/mnt/cti/subset.csv')
    parser.add_argument('--ntopic', default=10)
    parser.add_argument('--method', default='TFIDF')
    parser.add_argument('--savepng', default=0)
    # "TFIDF", "LDA", "BERT", "LDA_BERT"
    parser.add_argument('--samp_size', default=10000)
    args = parser.parse_args()

    data = pd.read_csv(str(args.fpath))
    data = data.fillna('')  # only the comments has NaN's
    rws = data.text
    sentences, token_lists, idx_in = preprocess(rws, samp_size=int(args.samp_size))
    # Define the topic model object
    tm = Topic_Model(k = int(args.ntopic), method = str(args.method))
    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)
    
    with open("/mnt/cti/docs/saved_models/{}-dictionary.pickle".format(tm.id), "wb") as f:
        pickle.dump(tm.dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/mnt/cti/docs/saved_models/{}-corpus.pickle".format(tm.id), "wb") as f:
        pickle.dump(tm.corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/mnt/cti/docs/saved_models/{}-cluster_model.pickle".format(tm.id), "wb") as f:
        pickle.dump(tm.cluster_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/mnt/cti/docs/saved_models/{}-ldamodel.pickle".format(tm.id), "wb") as f:
        pickle.dump(tm.ldamodel, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Evaluate using metrics
    with open("/mnt/cti/docs/saved_models/{}.pickle".format(tm.id), "wb") as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
    visualize(tm)
    
    if int(args.samp_size)==1:
        for i in range(tm.k):
            get_wordcloud(tm, token_lists, i)
