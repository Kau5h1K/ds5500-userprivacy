from yacs.config import CfgNode as CN
import os



########################################################################################################################
# INITIALIZE CONFIG
_C = CN()
########################################################################################################################
# DEFINE DATASET PARAMS

# PATH VARIABLES (INPUT)
_C.DATA = CN()
_C.DATA.INPUT = CN()
# root folder
_C.DATA.INPUT.ROOT_DPATH = os.path.join("..", "data", "OPP-115")
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
# master data for categorical models
_C.DATA.OUTPUT.CATMODEL_FPATH = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "master_catmodel_dataset.csv")
# categy-wise split annotations folder (w/o parsed JSON attr)
_C.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH  = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "catsplit_annotations_115_unparsed")
# categy-wise split annotations folder (w/ parsed JSON attr)
_C.DATA.OUTPUT.CATSPLIT_PARSED_DPATH  = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "catsplit_annotations_115_parsed")
# processed site metadata
_C.DATA.OUTPUT.SITE_METADATA_FPATH  = os.path.join(_C.DATA.OUTPUT.ROOT_DPATH, "site_metadata_115.csv")

# LABEL
_C.DATA.LABEL = "category"

########################################################################################################################
