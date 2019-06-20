
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/BacteriaClassifier.ipynb

import sys
sys.path.append("..")
from faigen.data import sequence
from faigen.data.sequence import regex_filter, count_filter, Dna2VecDataBunch
from functools import partial
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import manifold,neighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import plotly.plotly as py
import plotly.graph_objs as go
from fastai import *
from fastai.data_block import *
from fastai.basic_train import *
from fastai.layers import *
from fastai.metrics import *
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__=='__main__':

    print("Loading embedding")
    word_vectors = Word2Vec.load_word2vec_format('../faigen/pretrained/embeddings/dna2vec-20190611-1940-k8to8-100d-10c-4870Mbp-sliding-LmP.w2v')

    print("Loading Data")
    DB="/data/genomes/GenSeq_fastas"
    # DB='/home/serge/development/genomes/ncbi-genomes-2019-04-07/bacterial genomes'

    filters=[partial(regex_filter, rx="Bacillus|Staphylococcus|Vibrio|Rhizobium"),partial(regex_filter, rx="plasmid?\s", keep=False)]
    #        partial(count_filter,num_fastas=(1,1), keep=1)]

    bunch = Dna2VecDataBunch.from_folder(DB,test="test",
                 filters=filters,
                 labeler=lambda x: x.split()[1],
                 emb=word_vectors,ngram=8,skip=0,
                 n_cpus=7,agg=partial(np.mean, axis=0))

    print("Creating Learner")
    layers=[nn.Linear(bunch.train_dl.x.c,10),nn.ReLU(),
            nn.Linear(10,bunch.train_dl.y.c)]
    bac_classifier = SequentialEx(*layers)
    print(bac_classifier)
    learn = Learner(bunch, bac_classifier, metrics=[accuracy])

    print ("Training")
    learn.fit_one_cycle(3,5e-2)