from yacs.config import CfgNode as CN
import os
from pathlib import Path
import mlflow

########################################################################################################################
########################################################################################################################

# INITIALIZE CONFIG
_C = CN()

########################################################################################################################
########################################################################################################################

# DEFINE DATA PREPROCESS PATH VARIABLES

# PATH VARIABLES (INPUT)
_C.DATA = CN()
_C.DATA.INPUT = CN()
# root folder
_C.DATA.INPUT.ROOT_DPATH = (Path(__file__).resolve().parent.parent.parent / "OPP-115").as_posix()
# annotations folder
_C.DATA.INPUT.ANNOT_DPATH = os.path.join(_C.DATA.INPUT.ROOT_DPATH, "annotations")
# policy collection metadata file
_C.DATA.INPUT.POLCOL_METADATA_FPATH = os.path.join(_C.DATA.INPUT.ROOT_DPATH, "documentation", "policies_opp115.csv")
# Website metadata file
_C.DATA.INPUT.SITE_METADATA_FPATH = os.path.join(_C.DATA.INPUT.ROOT_DPATH, "documentation", "websites_opp115.csv")
# Sanitized policies folder
_C.DATA.INPUT.SANI_POL_DPATH  = os.path.join(_C.DATA.INPUT.ROOT_DPATH, "sanitized_policies")

# PATH VARIABLES (OUTPUT)
_C.DATA.OUTPUT = CN()
# processed data folder
_C.DATA.OUTPUT.ROOT_DPATH = os.path.join(_C.DATA.INPUT.ROOT_DPATH, "processed_data")
# processed annotation folder
_C.DATA.OUTPUT.ANNOT_DPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH , "processed_annotations")
# processed segments folder
_C.DATA.OUTPUT.SEGMENTS_DPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "processed_segments")
# master annotations file
_C.DATA.OUTPUT.ANNOT_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "master_annotations_115.csv")
# master data for categorical models (union)
_C.DATA.OUTPUT.CATMODEL_UNION_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "master_catmodel_dataset_union.csv")
# master data for categorical models with decoded cats (union)
_C.DATA.OUTPUT.CATMODEL_UNION_DECODED_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "master_catmodel_dataset_union_decoded.csv")
# master data for categorical models (majority vote)
_C.DATA.OUTPUT.CATMODEL_MAJORITY_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "master_catmodel_dataset_majority.csv")
# master data for categorical models with decoded cats (majority vote)
_C.DATA.OUTPUT.CATMODEL_MAJORITY_DECODED_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "master_catmodel_dataset_majority_decoded.csv")
# categy-wise split annotations folder (w/o parsed JSON attr)
_C.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH  = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "catsplit_annotations_115_unparsed")
# categy-wise split annotations folder (w/ parsed JSON attr)
_C.DATA.OUTPUT.CATSPLIT_PARSED_DPATH  = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "catsplit_annotations_115_parsed")
# processed site metadata
_C.DATA.OUTPUT.SITE_METADATA_FPATH  = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "site_metadata_115.csv")
# relational data for visualization
_C.DATA.OUTPUT.RDB_DPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "csv_relational_data")

########################################################################################################################
########################################################################################################################

# DEFINE DATASET ATTRIBUTES

# save files for visualization
_C.DATA.OUTPUT.SAVEFILE = True
# LABEL
_C.DATA.LABEL = "category"
# convert other category into its attributes
_C.DATA.ELEVATE_OTHER_ATTR = True

########################################################################################################################
########################################################################################################################

# DEFINE EMBEDDING PARAMS
_C.EMBED = CN()
# embeddings dir
_C.EMBED.OUTPUT_DPATH = os.path.join(_C.DATA.INPUT.ROOT_DPATH, "embeddings")
# corpus tokens and indices dictionary path
_C.EMBED.CORPUS_TOKEN_IDX_FPATH = os.path.join(_C.EMBED.OUTPUT_DPATH, "corpus_tokens_idx.pkl")

# glove embeddings folder
_C.EMBED.GLOVE_DPATH = os.path.join(_C.EMBED.OUTPUT_DPATH, "glove", "glove.6B")
# fastText embeddings folder
_C.EMBED.FASTTEXT_DPATH = os.path.join(_C.EMBED.OUTPUT_DPATH, "fastText")

########################################################################################################################
########################################################################################################################

# DEFINE GLOBAL, MODEL, and MISC PARAMS

_C.PARAM = CN()
_C.PARAM.DF_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "dataset.pkl")

_C.PARAM.SEED = 2021
_C.PARAM.DATASET = "majority"
_C.PARAM.CUDA = True
_C.PARAM.LOWER = True
_C.PARAM.STEM = False
_C.PARAM.TRAIN_SIZE = 0.7
_C.PARAM.CHAR_LEVEL = False
_C.PARAM.MAX_FILTER_SIZE = 1
_C.PARAM.BATCH_SIZE = 128
_C.PARAM.NUM_EPOCHS = 1
_C.PARAM.PATIENCE = 10
_C.PARAM.EMBED = None  # None, "glove", "fasttext", "domain"
_C.PARAM.FREEZE_EMBED = False
_C.PARAM.EMBED_DIM = 300
_C.PARAM.BEST_PARAM_DPATH = (Path(__file__).resolve().parent / "best_params").as_posix()
os.makedirs(_C.PARAM.BEST_PARAM_DPATH, exist_ok = True)

_C.PARAM.CUSTOM_PARAM_FPATH = (Path(__file__).resolve().parent / "custom_param_dict.json").as_posix()
########################################################################################################################
########################################################################################################################

# MLFLOW PARAMS

_C.MLFLOW = CN()
_C.MLFLOW.MODEL_REGISTRY = (Path(__file__).resolve().parent.parent.parent / "mlflow_registry").as_posix()
os.makedirs(_C.MLFLOW.MODEL_REGISTRY, exist_ok = True)
mlflow.set_tracking_uri("file://" + _C.MLFLOW.MODEL_REGISTRY)

########################################################################################################################
########################################################################################################################