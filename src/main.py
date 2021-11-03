import numpy as np
import warnings
import random
import torch

warnings.filterwarnings('ignore')

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.as_posix() # add project root path for jupyter/CLI
sys.path.insert(0, ROOT_DIR)
#sys.path.insert(0, "/Users/kaushik/MyStuff/Workspace/NEU/DS5500/Project/DS5500_CapstoneProject") # add project path for jupyter/CLI local
#sys.path.insert(0, "/home/kaushik/DS5500") # add project path for jupyter/CLI remote

from src.config import cfg
from src.data.PrivacyPolicyDataset import PrivacyPolicyDataset
from src.data.prepOPPCorpus import prepOPPCorpus
from src.utils import embeddings

def set_seeds(seed=2021):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

cuda = True
device = torch.device("cuda" if (
        torch.cuda.is_available() and cuda) else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")
if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("CUDA device is available. Setting default device to CUDA and default tensor type to cuda.FloatTensor")
else:
    print("CUDA device is not available. Setting default device to CPU and default tensor type to FloatTensor")


def main():
     set_seeds()
     prep_obj = prepOPPCorpus(cfg)
     prep_obj.processAnnotations(splitcat=True)
     prep_obj.preprocessSiteMetadata()
     prep_obj.createRelationalData()
     tokens_idx_dict = embeddings.Corpus2Tokens(cfg, read_pickle = True, clean = False)

     #dataset = PrivacyPolicyDataset(cfg)
     #X_train, X_dev, X_test, y_train, y_dev, y_test = dataset.splitData(test_size=0.20, dev_size=0.20, has_dev=True, is_majority = False)

if __name__ == '__main__':
    main()
