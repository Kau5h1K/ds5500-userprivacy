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
        self.dataset_majority, self.dataset_union = prep_obj.processAnnotations()
        self.metadata = prep_obj.preprocessSiteMetadata()


    def splitData(self, test_size=0.10, dev_size=0.10, has_dev=True, rand_state = seed, is_majority = True):
        """
        Function to split the dataset into train test or train-dev-test

        :param test_size: test set size
        :param dev_size: dev set size
        :param has_dev: if dev split is required, set True
        :param rand_state:
        :return: X_train, X_test, y_train, y_test or X_train, X_dev, X_test, y_train, y_dev, y_test
        """
        print("\nSplitting the {} dataset...".format("majority" if is_majority else "union" ))
        _X, _y = self.__X_and_y(is_majority)

        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=test_size, random_state=rand_state, stratify=_y)
        if has_dev:
            _X_train, _X_dev, _y_train, _y_dev = train_test_split(_X_train, _y_train, test_size=dev_size,
                                                                 random_state=rand_state, stratify=_y_train)
            print("-" * 60)
            print("Successfully split the {} dataset into {:g}% train, {:g}% dev and {:g}% test!".format("majority" if is_majority else "union" ,(1-dev_size-test_size)*100, dev_size*100, test_size*100))
            print("Number of unique segments in total: {}".format(_X.drop_duplicates().shape[0]))
            print("Number of rows in total: {}".format(len(_y)))
            self.splitStatistics(splitlist = [_X_train, _X_dev, _X_test, _y_train, _y_dev, _y_test], has_dev = True)
            return _X_train, _X_dev, _X_test, _y_train, _y_dev, _y_test
        else:
            print("-" * 60)
            print("Successfully split the {} dataset into {:g}% train and {:g}% test".format("majority" if is_majority else "union" , (1-test_size)*100, test_size*100))
            print("Number of unique segments in total: {}".format(_X.drop_duplicates().shape[0]))
            print("Number of rows in total: {}".format(len(_y)))
            self.splitStatistics(splitlist = [_X_train, _X_test, _y_train, _y_test], has_dev = False)
            return _X_train, _X_test, _y_train, _y_test


    def __X_and_y(self, is_majority = True):
        """
        Split data as X and y
        :return: data and label from df or pca
        """
        _X = None
        _y = None
        if is_majority:
            _X = self.dataset_majority.copy().drop(labels=[self.label], axis=1)
            _y = self.dataset_majority[self.label].copy()
        else:
            _X = self.dataset_union.copy().drop(labels=[self.label], axis=1)
            _y = self.dataset_union[self.label].copy()

        return _X, _y

    def decorate(func):
        """
        Function to decorate the output

        :param func: the function to wrap the output of
        :return: nested func
        """
        def inner(self, *args, **kwargs):
            print("~" * 60)
            func(self, *args, **kwargs)
            print("~" * 60)
        return inner

    @decorate
    def splitStatistics(self, splitlist, has_dev = True):
        if has_dev:
            _X_train, _X_dev, _X_test, _y_train, _y_dev, _y_test = splitlist
            for label, (X, y) in {"TRAIN SET":[_X_train, _y_train], "DEV SET":[_X_dev, _y_dev], "TEST SET":[_X_test, _y_test]}.items():
                print(label)
                print("Number of unique segments: {}".format(X.drop_duplicates().shape[0]))
                print("Number of rows: {}".format(len(y)))
                print("Percentage of segments containing each of the following categories:")
                df = _y_train.value_counts()
                print(pd.DataFrame({
                    "Counts": df,
                    "Percentage": (round(df/sum(df)*100, 2)).astype('str') + "%"}))
                print("-" * 60)



