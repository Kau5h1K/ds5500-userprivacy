import numpy as np
import pandas as pd
import random
import torch
import json
import pickle


def setDevice(cuda = True):
    """
    :param cuda: True if gpu usage is
    :return torch device (cpu or cuda)

    """
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device

def showDevice(device):
    if device.type == "cuda":
        print("🟢 CUDA device is available. Setting default device to CUDA and default tensor type to cuda.FloatTensor")
    else:
        print("🔴 CUDA device is not available. Setting default device to CPU and default tensor type to FloatTensor")


def setSeeds(seed=2021):
    """
    set seeds for reproducibility
    :param seed: random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def initParams():
    """
    initialize global param dictionary for modeling
    :return param dictionary
    """
    param_dict = {
    'dataset': None,
    "seed": None,
    "cuda": None,
    "lower": None,
    "stem": None,
    "train_size": None,
    "char_level": None,
    "max_filter_size": None,
    "batch_size": None,
    "embedding_dim": None,
    "num_filters": None,
    "hidden_dim": None,
    "dropout_p": None,
    "lr": None,
    "num_epochs": None,
    "embed": None,
    "freeze_embed": None,
    "patience": None,
    "threshold": None,
    "num_samples": None}

    return param_dict


def createParamDict(cfg):
    """
    configure and set initial params for modeling
    :param cfg: config
    :return initialized param dict
    """

    param_dict = initParams()
    param_dict['dataset'] = cfg.PARAM.DATASET
    param_dict['seed'] = cfg.PARAM.SEED
    param_dict['cuda'] = cfg.PARAM.CUDA
    param_dict['lower'] = cfg.PARAM.LOWER
    param_dict['stem'] = cfg.PARAM.STEM
    param_dict['train_size'] = cfg.PARAM.TRAIN_SIZE
    param_dict['char_level'] = cfg.PARAM.CHAR_LEVEL
    param_dict['max_filter_size'] = cfg.PARAM.MAX_FILTER_SIZE
    param_dict['batch_size'] = cfg.PARAM.BATCH_SIZE
    param_dict['num_epochs'] = cfg.PARAM.NUM_EPOCHS
    param_dict['patience'] = cfg.PARAM.PATIENCE
    param_dict["embed"] = cfg.PARAM.EMBED
    param_dict["freeze_embed"] = cfg.PARAM.FREEZE_EMBED
    param_dict["embedding_dim"] = cfg.PARAM.EMBED_DIM

    return param_dict


def loadDataset(cfg):
    """
    Load the dataset for modeling
    :param cfg: config
    :return loaded dataset (majority or union)
    """
    if cfg.PARAM.DATASET == "union":
        fpath = cfg.DATA.OUTPUT.CATMODEL_UNION_DECODED_FPATH
    else:
        fpath = cfg.DATA.OUTPUT.CATMODEL_MAJORITY_DECODED_FPATH

    dataset = pd.read_csv(fpath)
    dataset['category'] = dataset['category'].apply(lambda x: eval(x))

    try:
        with open(cfg.PARAM.DF_FPATH, "wb") as f:
            pickle.dump(dataset, f)
        print("🟢 Dataset loaded!")
    except:
        print("🔴 Dataset loading failed! Terminating...")
    return dataset


def loadID(fpath):
    """
    Load the text file for run ID
    :param fpath: file path for text file
    :return loaded run ID
    """
    with open(fpath) as f:
        run_ID = f.read().strip()
    return run_ID

def packageSegments(segments):
    counter = 1
    docs = []
    for segment in segments:
        docs.append({"content": segment, "meta": {"name": "Paragraph-{}".format(counter)}})
        counter+=1
    return docs

def loadParams(fpath):
    """
    Load the param dict for modeling
    :param fpath: file path for json file
    :return loaded param dict
    """
    with open(fpath) as f:
        params = json.load(f)
    return params


def saveParams(params, fpath, cls=None, sortkeys=False):
    """
    Save the param dict to local system
    :param params: param dict
    :param fpath: file path for json file
    :param cls: JSONEncoder
    :param sortkeys: sort the keys in JSON
    :return save param dict
    """
    with open(fpath, "w") as f:
        json.dump(params, indent=2, fp=f, cls=cls, sort_keys=sortkeys)


def savePickle(obj, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)


def loadPickle(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)


def rankSegments(list1, list2):
    assert(len(list1) == len(list2))
    if len(list1) == 0 or len(list2) == 0:
        return list1, list2
    list1, list2 = zip(*sorted(zip(list1, list2), reverse=True))
    return list(list1), list(list2)


def urlGen(url, start, end=None):
    if end is None:
        urlstr = url + "#:~:text=" + start
    else:
        urlstr = url + "#:~:text=" + start + "," + end
    return urlstr

def sitelistGen(segment_list, url, num_links = 3):
    sitelist_cat = []
    for segment in segment_list:
        urllist = []
        segment = segment.strip()
        segment_tokens = segment.split(" ")

        if len(segment_tokens) <= 5:
            start = segment
            gen_url = urlGen(url = url, start = start)
            urllist.append(gen_url)
        elif len(segment_tokens) > 5 and len(segment_tokens) <= 10:
            for i in range(num_links-1):
                start = " ".join(segment_tokens[0+i:2+i])
                end = " ".join(segment_tokens[::-1][0+i:2+i][::-1])
                gen_url = urlGen(url = url, start = start, end = end)
                urllist.append(gen_url)
        else:
            for i in range(num_links):
                start = " ".join(segment_tokens[0+i:3+i])
                end = " ".join(segment_tokens[::-1][0+i:3+i][::-1])
                gen_url = urlGen(url = url, start = start, end = end)
                urllist.append(gen_url)

        sitelist_cat.append(urllist)
    return sitelist_cat



if __name__ == '__main__':
    segments = loadPickle("segments.pkl")
    trigger = loadPickle("trigger.pkl")
    confidence = loadPickle("confidence.pkl")
    sitelists = loadPickle("sitelists.pkl")
    url = loadPickle("url.pkl")
    #sitelistGen(segments['User Choice/Control'], url, num_links = 3)






