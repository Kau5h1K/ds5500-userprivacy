import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from sklearn.model_selection import train_test_split
from src.utils import preprocess

seed = 2021
np.random.seed(seed)
random.seed(seed)


class PrivacyPolicyDataset:

    def __init__(self, cfg):

        self._cfg = cfg
        self.label = cfg.DATA.LABEL
        prep_obj = preprocess.PreprocessPrivacyPolicyDataset(cfg)
        self.dataset = prep_obj.processAnnotations()
        self.metadata = prep_obj.preprocessSiteMetadata()


    def splitData(self, test_size=0.10, dev_size=0.10, has_dev=False, rand_state = seed):
        """
        Function to split the dataset into train test or train-dev-test

        :param test_size: test set size
        :param dev_size: dev set size
        :param has_dev: if dev split is required, set True
        :param rand_state:
        :return: X_train, X_test, y_train, y_test or X_train, X_dev, X_test, y_train, y_dev, y_test
        """
        _X, _y = self.__X_and_y()

        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=test_size, random_state=rand_state)
        if has_dev:
            _X_train, _X_dev, _y_train, y_dev = train_test_split(_X_train, _y_train, test_size=dev_size,
                                                                 random_state=rand_state)
            return _X_train, _X_dev, _X_test, _y_train, y_dev, _y_test
        else:
            return _X_train, _X_test, _y_train, _y_test


    def __X_and_y(self):
        """
        Split data as X and y
        :return: data and label from df or pca
        """
        _X = None
        _y = None
        _X = self.df.copy().drop(labels=[self.label], axis=1)
        _y = self.df[self.label].copy()

        return _X, _y


