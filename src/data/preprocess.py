
import numpy as np
import pandas as pd
import torch
import itertools
import json
import re
from collections import Counter
from nltk.corpus import stopwords


from nltk.stem import PorterStemmer
from skmultilearn.model_selection import IterativeStratification




def cleanText(text, lower=True, stem=False, remove_stopwords=False, isolate_sym = True, remove_alphanum = False):
    """
    Function to clean the segment text
    :param text: text to process
    :param lower: lower the text
    :param stem: Perform stemming
    :param remove_stopwords: Remove stopwords
    :param isolate_sym: Isolate symbols
    :param remove_alphanum: Remove non-alphanumeric
    :return: cleaned segments text

    """

    # Lower text
    if lower:
        text = text.lower()

    # Remove stopwords
    if remove_stopwords:
        pattern = re.compile(r"\b(" + r"|".join(stopwords.words('english')) + r")\b\s*")
        text = pattern.sub("", text)

    # Isolate symbols
    if isolate_sym:
        text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)

    # Remove non-alphanumeric
    if remove_alphanum:
        text = re.sub("[^A-Za-z0-9]+", " ", text)

    # Remove Extra padding
    text = re.sub(" +", " ", text)
    text = text.strip()

    # Remove hyperlinks
    text = re.sub(r"http\S+", "", text)

    # Perform stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word) for word in text.split(" ")])

    return text



class LabelEncoder:
    """
    Class to encode/decode categories to one-hot encoding and viceversa
    Attribution: Code adapted from https://madewithml.com/
    """
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp) as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

    def __str__(self):
        return f"<MultiLabelLabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(list(itertools.chain.from_iterable(y)))
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        y_one_hot = np.zeros((len(y), len(self.class_to_index)), dtype=int)
        for i, item in enumerate(y):
            for class_ in item:
                y_one_hot[i][self.class_to_index[class_]] = 1
        return y_one_hot

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            indices = np.where(np.asarray(item) == 1)[0]
            classes.append([self.index_to_class[index] for index in indices])
        return classes


def train_test_split_multilabel(X, y, train_size=0.7, order = 1):
    """
    Train test split function for multi-label data
    :param X: X split
    :param y: y split
    :param train_size: size of train
    :param order: order of label combinations
    :return X_train, X_test, y_train, y_test
    """
    stratifier = IterativeStratification(n_splits=2, order=order, sample_distribution_per_fold=[1.0 - train_size, train_size])
    train_idx, test_idx = next(stratifier.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


class Tokenizer:
    """
    Class to tokenize the text on char/word level and obtain encoded sequences of text
    Attribution: Code adapted from https://madewithml.com
    """
    def __init__(self, char_level, num_tokens=None, pad_token="<PAD>", oov_token="<UNK>", token_to_index=None):
        self.char_level = char_level
        self.separator = "" if self.char_level else " "
        if num_tokens:
            num_tokens -= 2  # pad + unk tokens
        self.num_tokens = num_tokens
        self.pad_token = pad_token
        self.oov_token = oov_token
        if not token_to_index:
            token_to_index = {pad_token: 0, oov_token: 1}
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def __len__(self):
        return len(self.token_to_index)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if not self.char_level:
            texts = [text.split(" ") for text in texts]
        all_tokens = [token for text in texts for token in text]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        for token, count in counts:
            index = len(self)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            if not self.char_level:
                text = text.split(" ")
            sequence = []
            for token in text:
                sequence.append(self.token_to_index.get(token, self.token_to_index[self.oov_token]))
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for index in sequence:
                text.append(self.index_to_token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {
                "char_level": self.char_level,
                "oov_token": self.oov_token,
                "token_to_index": self.token_to_index,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp) as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

def Addpadding(lists, max_list_len = 0):

    """
    :param lists: input list of lists
    :param max_list_len: align all lists to this length with zeros
    """
    max_list_len = max(max_list_len, max(len(lst) for lst in lists))
    padded_lists = np.zeros((len(lists), max_list_len))
    for i, lst in enumerate(lists):
        padded_lists[i][:len(lst)] = lst
    return padded_lists