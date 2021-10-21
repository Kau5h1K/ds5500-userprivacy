import pandas as pd
import numpy as np
import glob
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
import os
import json
import re



class PreprocessPrivacyPolicyDataset:
    def __init__(self, cfg):

        self._cfg = cfg

    def processAnnotations(self):
        """
        Function to process the annotations

        :param self: the current instance of the class
        :return: master dataset for category model
        """
        print("Processing Annotations...")
        df_list = []

        for fname in glob.glob(r"{}/*.csv".format(self._cfg.DATA.INPUT.ANNOT_DPATH)):

            #Extract path basename
            basename = os.path.basename(fname)

            #Create directories if they don't exist
            os.makedirs(self._cfg.DATA.OUTPUT.ANNOT_DPATH, exist_ok = True)
            os.makedirs(self._cfg.DATA.OUTPUT.SEGMENTS_DPATH, exist_ok = True)

            #Extract policyID from basename
            policy_id = basename.split('_')[0]
            policy_df = pd.read_csv(fname, header=None, usecols=[0, 2, 4, 5, 6], names=['annotation_ID','annotator_ID', 'segment_ID', 'category', 'attr_val'])

            #Set policyID in each table
            policy_df.loc[:,"policy_ID"] = policy_id

            #Replace extension
            santized_policy_fpath = os.path.splitext(basename)[0]+'.html'

            # Parse html text
            html = open(os.path.join(self._cfg.DATA.INPUT.SANI_POL_DPATH, santized_policy_fpath), "r").read()
            soup = BeautifulSoup(html, features="html.parser")
            soup_text = soup.get_text()

            #Match segments with their segment IDs for each policy
            segments = soup_text.split("|||")
            segments_df = pd.DataFrame(segments, columns = ["segment_text"])
            segments_df.index.name = "segment_ID"
            segments_df.reset_index(inplace = True)

            #Save processed segments
            segments_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.SEGMENTS_DPATH, basename), index = False)

            policy_df_merged = policy_df.merge(segments_df, on='segment_ID', how = "inner")

            #Save processed policies
            policy_df_merged.to_csv(os.path.join(self._cfg.DATA.OUTPUT.ANNOT_DPATH, basename), index = False)

            df_list.append(policy_df_merged)

        master_annotations_df = pd.concat(df_list, axis=0, ignore_index=True)
        assert(master_annotations_df.isnull().any(axis=1).any() == False)

        if self._cfg.DATA.ELEVATE_OTHER_ATTR:
            master_annotations_df = self.elevateOtherCategoryAttr(master_annotations_df)

        master_annotations_df.to_csv(self._cfg.DATA.OUTPUT.ANNOT_FPATH, index = False)

        cat_model_dataset_union = master_annotations_df[["segment_text", "category"]].drop_duplicates()
        cat_model_dataset_majority = self.createMajorityDataset(master_annotations_df[["policy_ID", "segment_ID", "annotator_ID", "segment_text", "category"]].drop_duplicates())


        cat_model_dataset_majority.to_csv(self._cfg.DATA.OUTPUT.CATMODEL_MAJORITY_FPATH, index = False)
        cat_model_dataset_union.to_csv(self._cfg.DATA.OUTPUT.CATMODEL_UNION_FPATH, index = False)

        print("Processing annotations complete!")
        print("Saved Processed segments to directory {}".format(self._cfg.DATA.OUTPUT.SEGMENTS_DPATH))
        print("Saved Processed annotations to directory {}".format(self._cfg.DATA.OUTPUT.ANNOT_DPATH))
        print("Saved master annotation to file {}".format(self._cfg.DATA.OUTPUT.ANNOT_FPATH))
        print("Saved master dataset for category model (majority) to file {}".format(self._cfg.DATA.OUTPUT.CATMODEL_MAJORITY_FPATH))
        print("Saved master dataset for category model (union) to file {}".format(self._cfg.DATA.OUTPUT.CATMODEL_UNION_FPATH))
        print("-"*60)

        self.splitCategories(master_annotations_df)

        return cat_model_dataset_majority, cat_model_dataset_union

    def splitCategories(self, annotation_df):
        """
        Function to split the dataset by categories

        :param self: the current instance of the class
        :param annotation_df: merged annotation file of all policies
        """
        print("Splitting annotations by categories ...")
        cat_dfs_list = [df for _, df in annotation_df.groupby('category')]

        #Directory exist check
        os.makedirs(self._cfg.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH, exist_ok = True)

        #Save them into corresponding .CSV files
        for df in cat_dfs_list:
            assert(len(set(df.category)) == 1)
            category = '_'.join(next(iter(set(df.category))).replace("/", "-").split(" "))
            df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH, "{}.csv".format(category)), index = False)
        print("Splitting annotations complete!")
        print("Saved unparsed category-wise split annotations to directory {}".format(self._cfg.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH))
        print("-"*60)
        self.parseAttr(cat_dfs_list)

    def parseAttr(self, cat_dfs_list):
        """
        Function to parse the attr key value pairs (JSON)

        :param self: the current instance of the class
        :param cat_dfs_list: list of dataframes wrt to each category (len of 10)
        """
        print("Parsing attribute key-value pairs (JSON)...")
        for policy_df in cat_dfs_list:
            policy_df.reset_index(inplace=True, drop=True)
            assert(len(set(policy_df.category)) == 1)
            category = '_'.join(next(iter(set(policy_df.category))).replace("/", "-").split(" "))
            os.makedirs(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH, exist_ok = True)

            cat_list_of_dict = []
            for index, row in policy_df.iterrows():
                attr_dict = json.loads(row["attr_val"])
                cat_list_of_dict.append({ k:v['value'] for k,v in attr_dict.items() })
            cat_df = pd.DataFrame(cat_list_of_dict)
            policy_df.drop(["attr_val"], axis = 1, inplace = True)
            assert(cat_df.isnull().any(axis=1).any() == False)
            pd.concat((policy_df, cat_df), axis = 1).to_csv(os.path.join(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH, "{}.csv".format(category)), index = False)
        print("Parsing complete!")
        print("Saved parsed category-wise split annotations to directory {}".format(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH))

    def preprocessSiteMetadata(self):
        """
        Function to process the site metadata

        :param self: the current instance of the class
        :return: site metadata file
        """
        print("Processing site metadata...!")
        site_metadata_df = pd.read_csv(self._cfg.DATA.INPUT.SITE_METADATA_FPATH)
        # manually added a us rank of 0 to a missing value for policy UID 745

        alexa_rank_global = []
        alexa_rank_us = []
        sectors = []

        for index, row in site_metadata_df.iterrows():
            sector_lst = []
            alexa_rank_global.append(re.findall(r'\d+', row["Comments"])[0])
            alexa_rank_us.append(re.findall(r'\d+', row["Comments"])[1])

            sector_lst = list(set([row.iloc[i].split(":")[0] for i in range(7, site_metadata_df.shape[1]) if not row.iloc[i] != row.iloc[i]]))
            sectors.append(sector_lst)

        metadata_df = pd.DataFrame({ 'site_name': site_metadata_df["Site Human-Readable Name"].values,
                       'policy_ID': site_metadata_df["Policy UID"].values,
                       'alexa_rank_global': alexa_rank_global,
                       'alexa_rank_us': alexa_rank_us,
                       'sectors': sectors
                       })
        metadata_df.to_csv(self._cfg.DATA.OUTPUT.SITE_METADATA_FPATH, index = False)
        print("Processing metadata complete!")
        print("Saved site metadata to file {}".format(self._cfg.DATA.OUTPUT.SITE_METADATA_FPATH))
        return metadata_df


    def createMajorityDataset(self, dataset):
        dataset.segment_ID = dataset.segment_ID.astype('str')
        dataset.policy_ID = dataset.policy_ID.astype('str')
        dataset['polID_segID_cat'] = dataset[['policy_ID','segment_ID', 'category']].agg('-'.join, axis=1)
        dataset_counts = dataset.groupby('polID_segID_cat').nunique()['annotator_ID']
        selected_rows = dataset_counts[dataset_counts >= 2].index
        majority_data = dataset[dataset.polID_segID_cat.isin(selected_rows)][["segment_text", "category"]].drop_duplicates().reset_index(drop=True)
        return majority_data

    def elevateOtherCategoryAttr(self, df):

        for i,r in df.iterrows():
            if r['category'] == "Other":
                attr_dict = json.loads(r["attr_val"])
                df.iloc[i, 3] = attr_dict["Other Type"]["value"]
        df = df[df.category != "Other"]
        return df