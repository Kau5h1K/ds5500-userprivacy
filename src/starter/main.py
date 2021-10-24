import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings('ignore')

from src.config import cfg
from src.data.PrivacyPolicyDataset import PrivacyPolicyDataset
from src.utils import preprocess


def main():
     dataset = PrivacyPolicyDataset(cfg)
     X_train, X_dev, X_test, y_train, y_dev, y_test = dataset.splitData(test_size=0.20, dev_size=0.20, has_dev=True, is_majority = False)
     print("")

if __name__ == '__main__':
    main()
