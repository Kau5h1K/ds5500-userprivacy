import pandas as pd
import numpy as np
import glob
import warnings
import torch
warnings.filterwarnings('ignore')

from src.config import cfg
from src.data.PrivacyPolicyDataset import PrivacyPolicyDataset
from src.utils.preprocess import PreprocessPrivacyPolicyDataset
from src.utils import embeddings



def main():
     prep_obj = PreprocessPrivacyPolicyDataset(cfg)
     prep_obj.processAnnotations(splitcat=False)
     #prep_obj.preprocessSiteMetadata()
     #prep_obj.createRelationalData()
     tokens_idx_dict = embeddings.Corpus2Tokens(cfg, read_pickle = True, clean = False)

     dataset = PrivacyPolicyDataset(cfg)
     #X_train, X_dev, X_test, y_train, y_dev, y_test = dataset.splitData(test_size=0.20, dev_size=0.20, has_dev=True, is_majority = False)

if __name__ == '__main__':
    main()
