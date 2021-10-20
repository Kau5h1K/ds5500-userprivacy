from yacs.config import CfgNode as CN
import os



########################################################################################################################
# INITIALIZE CONFIG
_C = CN()
########################################################################################################################
# DEFINE DATASET PARAMS

# PATH VARIABLES
_C.DATA = CN()
_C.DATA.ROOT_DPATH = os.path.join("..", "data", "OPP-115")
_C.DATA.ANNOT_DPATH = os.path.join(_C.DATA.ROOT_DPATH, "annotations")
_C.DATA.POLCOL_METADATA_FPATH = os.path.join(_C.DATA.ROOT_DPATH, "documentation", "policies_opp115.csv")
_C.DATA.SITE_METADATA_FPATH = os.path.join(_C.DATA.ROOT_DPATH, "documentation", "websites_opp115.csv")
_C.DATA.SANI_POL_DPATH  = os.path.join(_C.DATA.ROOT_DPATH, "sanitized_policies")

# LABEL
_C.DATA.LABEL = "category"

########################################################################################################################
