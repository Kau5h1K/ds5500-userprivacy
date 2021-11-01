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
        if cfg.DATA.IS_UNION:
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




