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


if __name__ == '__main__':
    main()
