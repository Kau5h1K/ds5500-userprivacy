import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import pickle
from collections import OrderedDict
import torch
from src.config import cfg
from gensim.models import FastText, fasttext
from gensim.test.utils import datapath

def cleanText(t):
    """
        Clean input text
        :param: t: text string
        :return: cleaned text
    """
    encoded_string = t.encode("ascii", "ignore")
    t = encoded_string.decode()
    t = re.sub(r'(www).*[\s]*', '', t)
    t = re.sub(r"[^A-Za-z0-9,!?*.;’´'\/]", " ", t)
    t = re.sub(r",", " ", t)
    t = re.sub(r"’", "'", t)
    t = re.sub(r"´", "'", t)
    t = re.sub(r"\.", " ", t)
    t = re.sub(r"!", " ! ", t)
    t = re.sub(r"\?", " ? ", t)
    t = re.sub(r"\/", " ", t)
    return t


def Corpus2Tokens(cfg, read_pickle = True, clean = False):
    """
        Convert OPP-115 corpus into a dictionary of tokens with indices
        :param: cfg: config variable
        :param: read_pickle: read from saved pickle object
        :return: dictionary with keys and values as words and indices
    """
    try:
        if not read_pickle:
            raise()

        print("Loading the token index dictionary from file, {}".format(cfg.EMBED.CORPUS_TOKEN_IDX_FPATH))
        with open(cfg.EMBED.CORPUS_TOKEN_IDX_FPATH,"rb") as f:
            corpus_tokens_idx = pickle.load(f)

    except:
        print("Creating the token index dictionary with filename, {}".format(cfg.EMBED.CORPUS_TOKEN_IDX_FPATH))
        os.makedirs(cfg.EMBED.OUTPUT_DPATH, exist_ok = True)
        if cfg.PARAM.DATASET == "union":
            corpus_path = cfg.DATA.OUTPUT.CATMODEL_UNION_FPATH
        else:
            corpus_path = cfg.DATA.OUTPUT.CATMODEL_MAJORITY_FPATH

        corpus_df = pd.read_csv(corpus_path)

        token_set = set()
        for i, r in corpus_df.iterrows():
            segment = corpus_df.iloc[i,0]
            if clean:
                segment = cleanText(segment)
            token_set = token_set.union({token.lower() for token in nltk.word_tokenize(segment)})

        token_list = sorted(token_set)

        corpus_tokens_idx = {None: 0}

        for idx, token in enumerate(token_list,1):

            corpus_tokens_idx[token] = idx

        with open(cfg.EMBED.CORPUS_TOKEN_IDX_FPATH, "wb") as f:

            pickle.dump(corpus_tokens_idx, f)

    return corpus_tokens_idx

def loadfastText(fpath, word_idx):
    """
        Load fastText embeddings file to a dictionary
        :param: fpath: path of GloVe word embeddings file
        :return: dictionary with keys and values as words and embeddings
    """
    print("Loading fasttext embeddings file to a dictionary...")
    embeddings = {}
    embeddings_model = fasttext.load_facebook_model(fpath)
    for word, _ in word_idx.items():
        embedding = np.asarray(embeddings_model.wv[word])
        embeddings[word] = embedding

    return embeddings


def loadGloVe(fpath):
    """
        Load glove embeddings file to a dictionary
        :param: fpath: path of GloVe word embeddings file
        :return: dictionary with keys and values as words and embeddings
    """
    print("Loading GloVe embeddings file to a dictionary...")
    embeddings = {}
    with open(fpath, "r") as f:
        for index, line in enumerate(f):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings


def createEmbeddingMatrix(embeddings, word_idx, embed_dim):
    """
        m
        :param: fpath: path of GloVe word embeddings file
        :return: dictionary with keys and values as words and embeddings
    """
    embedding_mat = np.zeros((len(word_idx), embed_dim))
    for w, i in word_idx.items():
        embedding_vec = embeddings.get(w)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    return embedding_mat


def processEmbeddings(params, tokenizer):
    if params.embed is None:
        return None
    elif params.embed == "glove":
        embeddings_fpath = os.path.join(cfg.EMBED.GLOVE_DPATH, "embed_mat_glove_{}.pkl".format(params.embedding_dim))
        try:
            print("Searching for GloVe embedding matrix in cached reserves")
            with open(embeddings_fpath, "rb") as f:
                embedding_mat = pickle.load(f)
        except:
            print("Cached GloVe Embedding matrix not found. Creating new one...")
            embeddings_fpath_ip = os.path.join(cfg.EMBED.GLOVE_DPATH, 'glove.6B.{}d.txt'.format(params.embedding_dim))
            glove_embed = loadGloVe(fpath=embeddings_fpath_ip)
            embedding_mat = createEmbeddingMatrix(embeddings=glove_embed, word_idx = tokenizer.token_to_index, embed_dim = params.embedding_dim)
            print (f" Created GloVe <Embeddings(words={embedding_mat.shape[0]}, dim={embedding_mat.shape[1]})>")
            with open(embeddings_fpath, "wb") as f:
                pickle.dump(embedding_mat, f)
        return embedding_mat
    elif params.embed == "fasttext":
        embeddings_fpath = os.path.join(cfg.EMBED.FASTTEXT_DPATH, "embed_mat_fasttext_300.pkl")
        try:
            print("Searching for fastText embedding matrix in cached reserves")
            with open(embeddings_fpath) as f:
                embedding_mat = pickle.load(f)
        except:
            print("Cached fastText Embedding matrix not found. Creating new one...")
            embeddings_fpath_ip = os.path.join(cfg.EMBED.FASTTEXT_DPATH, "cc.en.300.bin")
            fasttext_embed = loadfastText(fpath=embeddings_fpath_ip, word_idx = tokenizer.token_to_index)
            embedding_mat = createEmbeddingMatrix(embeddings=fasttext_embed, word_idx = tokenizer.token_to_index, embed_dim = params.embedding_dim)
            print (f" Created GloVe <Embeddings(words={embedding_mat.shape[0]}, dim={embedding_mat.shape[1]})>")
            with open(embeddings_fpath, "wb") as f:
                pickle.dump(embedding_mat, f)
        return embedding_mat
    elif params.embed == "domain":
        pass