import pandas as pd
import numpy as np
import glob
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
import os
import json
import re
from collections import OrderedDict
from src.utils import gen


class prepOPPCorpus:
    """ Prepares dataset for VIZ and ML from raw OPP-115 corpus
    """
    def __init__(self, cfg):

        self._cfg = cfg
        self.cat2idx = OrderedDict([('First Party Collection/Use', 0),
                                   ('Third Party Sharing/Collection', 1),
                                   ('User Access, Edit and Deletion', 2),
                                   ('Data Retention', 3),
                                   ('Data Security', 4),
                                   ('International and Specific Audiences', 5),
                                   ('Do Not Track', 6),
                                   ('Policy Change', 7),
                                   ('User Choice/Control', 8),
                                   ('Introductory/Generic', 9),
                                   ('Practice not covered', 10),
                                   ('Privacy contact information', 11)])
        self.idx2cat = OrderedDict([(0, 'First Party Collection/Use'),
                                    (1, 'Third Party Sharing/Collection'),
                                    (2, 'User Access, Edit and Deletion'),
                                    (3, 'Data Retention'),
                                    (4, 'Data Security'),
                                    (5, 'International and Specific Audiences'),
                                    (6, 'Do Not Track'),
                                    (7, 'Policy Change'),
                                    (8, 'User Choice/Control'),
                                    (9, 'Introductory/Generic'),
                                    (10, 'Practice not covered'),
                                    (11, 'Privacy contact information')])

    def processAnnotations(self, splitcat = False):
        """
        Function to process the annotations

        :param self: the current instance of the class
        :param splitcat: Process datasaet for visualization (split the dataset wrt categories
        :return: master dataset for category model
        """
        print("Processing Annotations...")
        df_list = []

        for fname in glob.glob(r"{}/*.csv".format(self._cfg.DATA.INPUT.ANNOT_DPATH)):

            #Extract path basename
            basename = os.path.basename(fname)

            #Create directories if they don't exist
            if self._cfg.DATA.OUTPUT.SAVEFILE:
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
            if self._cfg.DATA.OUTPUT.SAVEFILE:
                segments_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.SEGMENTS_DPATH, basename), index = False)

            policy_df_merged = policy_df.merge(segments_df, on='segment_ID', how = "inner")

            #Save processed policies
            if self._cfg.DATA.OUTPUT.SAVEFILE:
                policy_df_merged.to_csv(os.path.join(self._cfg.DATA.OUTPUT.ANNOT_DPATH, basename), index = False)

            df_list.append(policy_df_merged)

        master_annotations_df = pd.concat(df_list, axis=0, ignore_index=True)
        assert(master_annotations_df.isnull().any(axis=1).any() == False)

        if self._cfg.DATA.ELEVATE_OTHER_ATTR:
            master_annotations_df = self.elevateOtherCategoryAttr(master_annotations_df)

        if self._cfg.DATA.OUTPUT.SAVEFILE:
            master_annotations_df.to_csv(self._cfg.DATA.OUTPUT.ANNOT_FPATH, index = False)

        cat_model_dataset_union, cat_model_dataset_union_decoded = self.createUnionDataset(master_annotations_df)
        cat_model_dataset_majority, cat_model_dataset_majority_decoded = self.createMajorityDataset(master_annotations_df)

        if self._cfg.DATA.OUTPUT.SAVEFILE:
            cat_model_dataset_majority.to_csv(self._cfg.DATA.OUTPUT.CATMODEL_MAJORITY_FPATH, index = False)
            cat_model_dataset_majority_decoded.to_csv(self._cfg.DATA.OUTPUT.CATMODEL_MAJORITY_DECODED_FPATH, index = False)
            cat_model_dataset_union.to_csv(self._cfg.DATA.OUTPUT.CATMODEL_UNION_FPATH, index = False)
            cat_model_dataset_union_decoded.to_csv(self._cfg.DATA.OUTPUT.CATMODEL_UNION_DECODED_FPATH, index = False)

        print("Processing annotations complete!")
        if self._cfg.DATA.OUTPUT.SAVEFILE:
            print("Saved Processed segments to directory {}".format(self._cfg.DATA.OUTPUT.SEGMENTS_DPATH))
            print("Saved Processed annotations to directory {}".format(self._cfg.DATA.OUTPUT.ANNOT_DPATH))
            print("Saved master annotation to file {}".format(self._cfg.DATA.OUTPUT.ANNOT_FPATH))
            print("Saved master dataset for category model (majority) to file {}".format(self._cfg.DATA.OUTPUT.CATMODEL_MAJORITY_FPATH))
            print("Saved master dataset for category model (union) to file {}".format(self._cfg.DATA.OUTPUT.CATMODEL_UNION_FPATH))
        print("-"*60)

        if splitcat:
            self.splitCategories(master_annotations_df)


    def splitCategories(self, annotation_df):
        """
        Function to split the dataset by categories

        :param self: the current instance of the class
        :param annotation_df: merged annotation file of all policies
        """
        print("Splitting annotations by categories ...")
        cat_dfs_list = [df for _, df in annotation_df.groupby('category')]

        #Directory exist check
        if self._cfg.DATA.OUTPUT.SAVEFILE:
            os.makedirs(self._cfg.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH, exist_ok = True)

        #Save them into corresponding .CSV files
        if self._cfg.DATA.OUTPUT.SAVEFILE:
            for df in cat_dfs_list:
                assert(len(set(df.category)) == 1)
                category = '_'.join(next(iter(set(df.category))).replace("/", "-").split(" "))
                df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.CATSPLIT_UNPARSED_DPATH, "{}.csv".format(category)), index = False)
        print("Splitting annotations complete!")
        if self._cfg.DATA.OUTPUT.SAVEFILE:
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
            if self._cfg.DATA.OUTPUT.SAVEFILE:
                os.makedirs(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH, exist_ok = True)

            cat_list_of_dict = []
            for index, row in policy_df.iterrows():
                attr_dict = json.loads(row["attr_val"])
                cat_list_of_dict.append({ k:v['value'] for k,v in attr_dict.items() })
            cat_df = pd.DataFrame(cat_list_of_dict)
            policy_df.drop(["attr_val"], axis = 1, inplace = True)
            assert(cat_df.isnull().any(axis=1).any() == False)
            final_df = pd.concat((policy_df, cat_df), axis = 1)
            if self._cfg.DATA.OUTPUT.SAVEFILE:
                final_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH, "{}.csv".format(category)), index = False)
        print("Parsing complete!")
        if self._cfg.DATA.OUTPUT.SAVEFILE:
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
        if self._cfg.DATA.OUTPUT.SAVEFILE:
            metadata_df.to_csv(self._cfg.DATA.OUTPUT.SITE_METADATA_FPATH, index = False)
        print("Processing metadata complete!")
        if self._cfg.DATA.OUTPUT.SAVEFILE:
            print("Saved site metadata to file {}".format(self._cfg.DATA.OUTPUT.SITE_METADATA_FPATH))

    def encodeCategories(self, category, catIndex):
        """
        Function to encode categories to binary

        :param category: current category
        :param catIndex: an OrderedDict to choose the encoding map from
        :return: encoded list
        """
        cat_encoded = np.zeros((len(catIndex)))
        cat_encoded[catIndex[category]] = 1
        return cat_encoded

    def decode_labels(self, x, labels_map):
        """
        Function to decode binary categories to original text

        :param x: current category
        :param labels_map: an OrderedDict to choose the decoding map from
        :return: decoded list
        """
        lst = []
        for i, lab in enumerate(x):
            if lab == 1:
                lst.append(labels_map[i])
        return lst

    def createUnionDataset(self, data):
        """
        Function to create union gold standard dataset

        :param data: dataset with all annotations
        :return: union dataset
        """
        dataset = data[["policy_ID", "segment_ID", "segment_text", "category"]].drop_duplicates().reset_index(drop=True)
        union_data = pd.DataFrame({'segment_text': [], 'category': []})
        g = dataset.groupby(["policy_ID"])
        for _, df in g:
            df['category'] = df['category'].apply(lambda x: self.encodeCategories(x, self.cat2idx))
            encoded_cats = df[['segment_ID', 'category']].groupby("segment_ID").sum()
            uniq_segments = df[['segment_ID', 'segment_text']].set_index('segment_ID').drop_duplicates()
            encoded_df = pd.merge(uniq_segments, encoded_cats, left_index=True, right_index=True)
            union_data = pd.concat([union_data, encoded_df])

        union_data["category"] = union_data["category"].apply(lambda x: list(x))
        union_data.reset_index(drop=True, inplace=True)
        union_data_decoded = union_data.copy()
        union_data_decoded["category"] = union_data_decoded["category"].apply(lambda x: self.decode_labels(x, self.idx2cat))
        return union_data, union_data_decoded

    def createMajorityDataset(self, data):
        """
        Function to create majority gold standard dataset

        :param data: dataset with all annotations
        :return: majority dataset
        """
        dataset = data[["policy_ID", "segment_ID", "annotator_ID", "segment_text", "category"]].drop_duplicates()
        dataset.segment_ID = dataset.segment_ID.astype('str')
        dataset.policy_ID = dataset.policy_ID.astype('str')
        dataset['polID_segID_cat'] = dataset[['policy_ID','segment_ID', 'category']].agg('-'.join, axis=1)
        dataset_counts = dataset.groupby('polID_segID_cat').nunique()['annotator_ID']
        selected_rows = dataset_counts[dataset_counts >= 2].index
        dataset = dataset[dataset.polID_segID_cat.isin(selected_rows)][["policy_ID", "segment_ID", "segment_text", "category"]].drop_duplicates().reset_index(drop=True)

        majority_data = pd.DataFrame({'segment_text': [], 'category': []})
        g = dataset.groupby(["policy_ID"])
        for _, df in g:
            df['category'] = df['category'].apply(lambda x: self.encodeCategories(x, self.cat2idx))
            encoded_cats = df[['segment_ID', 'category']].groupby("segment_ID").sum()
            uniq_segments = df[['segment_ID', 'segment_text']].set_index('segment_ID').drop_duplicates()
            encoded_df = pd.merge(uniq_segments, encoded_cats, left_index=True, right_index=True)
            majority_data = pd.concat([majority_data, encoded_df])

        majority_data["category"] = majority_data["category"].apply(lambda x: list(x))
        majority_data.reset_index(drop=True, inplace=True)
        majority_data_decoded = majority_data.copy()
        majority_data_decoded["category"] = majority_data_decoded["category"].apply(lambda x: self.decode_labels(x, self.idx2cat))

        return majority_data, majority_data_decoded

    def elevateOtherCategoryAttr(self, df):
        """
        Function to elevate Other category attributes to category level

        :param df: dataset with all annotations
        :return modified df
        """

        for i,r in df.iterrows():
            if r['category'] == "Other":
                attr_dict = json.loads(r["attr_val"])
                df.iloc[i, 3] = attr_dict["Other Type"]["value"]
        df = df[df.category != "Other"]
        return df

    def genIDs(self, num):
        """
        Function to generate primary key IDs for new tables

        :param num: number of IDs required
        :return: list of IDS
        """
        id_lst = []
        for i in range(num):
            id_lst.append(i)
        return id_lst

    def createRelationalData(self):
        """
        Function to create Relational data files for VIZ

        :return: list of .CSV files saved to file system
        """
        site_metadata_df = pd.read_csv(self._cfg.DATA.OUTPUT.SITE_METADATA_FPATH)
        lst = []
        for i,r in site_metadata_df.iterrows():
            for j in eval(r['sectors']):
                lst.append(j)
        unique_sec = list(set(lst))
        site_metadata_df_orig = site_metadata_df.copy()
        for sec in unique_sec:
            site_metadata_df[sec] = 0
        for i,r in site_metadata_df.iterrows():
            for sec in eval(r['sectors']):
                site_metadata_df.loc[i, sec] = 1
        site_metadata_df.drop(['sectors'], axis = 1, inplace = True)
        site_sector_map_df = pd.melt(site_metadata_df, id_vars=['policy_ID'], value_vars= site_metadata_df.columns[4:],
                                     var_name='sector', value_name='sector_val')
        site_sector_map_df = site_sector_map_df[site_sector_map_df.sector_val == 1]
        site_sector_map_df = site_sector_map_df.drop(["sector_val"], axis = 1).reset_index(drop = True)
        ids = self.genIDs(site_sector_map_df.shape[0])
        site_sector_map_df['site_sector_map_ID'] = ids
        site_sector_map_df = site_sector_map_df[["site_sector_map_ID", "policy_ID", "sector"]]
        site_metadata_df_orig.drop(["sectors"], inplace = True, axis = 1)


        master_df = pd.read_csv(self._cfg.DATA.OUTPUT.ANNOT_FPATH)
        lst_cat = list(set(master_df.category))
        ids = self.genIDs(len(lst_cat))
        cat_metadata_df = pd.DataFrame(list(zip(ids, lst_cat)), columns = ["category_ID", "category"])


        master_df = pd.read_csv(self._cfg.DATA.OUTPUT.ANNOT_FPATH)
        master_df = master_df.merge(cat_metadata_df, how = "inner", on = "category")
        master_df.drop(["category", "attr_val"], axis = 1, inplace = True)


        cat_attr_lst = []
        for fname in glob.glob(r"{}/*.csv".format(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH)):
            if not (os.path.basename(fname) in ["Introductory-Generic.csv", "Practice_not_covered.csv", "Privacy_contact_information.csv"]):
                df = pd.read_csv(fname)
                cat_attr_lst.append({col:next(iter(set(df.category))) for col in list(df.columns[6:])})


        attr_cat_map_df = pd.DataFrame(
            [{"attr": key, "category": value} for d in cat_attr_lst for key, value in d.items() ])
        ids = self.genIDs(attr_cat_map_df.shape[0])
        attr_cat_map_df['attr_cat_map_ID'] = ids
        attr_cat_map_df = attr_cat_map_df[["attr_cat_map_ID", "category", "attr"]]

        attr_table_df = attr_cat_map_df[["attr"]].drop_duplicates(ignore_index=True)
        ids = self.genIDs(attr_table_df.shape[0])
        attr_table_df['attr_ID'] = ids
        attr_table_df = attr_table_df[["attr_ID", "attr"]]

        attr_cat_map_df = attr_cat_map_df.merge(attr_table_df, how = "inner", on = "attr").merge(cat_metadata_df, how = "inner", on = "category")
        attr_cat_map_df = attr_cat_map_df[["attr_cat_map_ID", "category_ID", "attr_ID"]]


        attr_data_df = pd.DataFrame()
        for fname in glob.glob(r"{}/*.csv".format(self._cfg.DATA.OUTPUT.CATSPLIT_PARSED_DPATH)):
            df = pd.read_csv(fname)
            attr_data_df = pd.concat([attr_data_df, pd.melt(df, id_vars=['annotation_ID'], value_vars= df.columns[6:],
                                                            var_name='attr', value_name='attr_val')], axis =0, ignore_index = True)

        attr_data_df = attr_data_df.merge(attr_table_df, how = "inner", on = "attr")
        ids = self.genIDs(attr_data_df.shape[0])
        attr_data_df['attr_data_ID'] = ids
        attr_data_df = attr_data_df[["attr_data_ID", "annotation_ID", "attr_ID", "attr_val"]]

        if self._cfg.DATA.OUTPUT.SAVEFILE:
            os.makedirs(self._cfg.DATA.OUTPUT.RDB_DPATH, exist_ok = True)
            site_metadata_df_orig.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "site_metadata.csv"), index = False)
            site_sector_map_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "site_sector_map.csv"), index = False)
            cat_metadata_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "category_metadata.csv"), index = False)
            master_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "annotation_data.csv"), index = False)
            attr_table_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "attr_metadata.csv"), index = False)
            attr_cat_map_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "attr_category_map.csv"), index = False)
            attr_data_df.to_csv(os.path.join(self._cfg.DATA.OUTPUT.RDB_DPATH, "attr_values_map.csv"), index = False)




def createDataset(cfg, splitcat = False, metadata = False, relational_data = False):
    """
    Top-level Function to create Datasets for ML and VIZ

    :param cfg: config
    :param splitcat: Split the categories (calls splitCategories func)
    :param metadata: Create metadata file (calls preprocessSiteMetadata func)
    :param relational_data: creates relational data (calls createRelationalData func)
    """
    gen.setSeeds(cfg.PARAM.SEED)
    prep_obj = prepOPPCorpus(cfg)
    prep_obj.processAnnotations(splitcat)
    if metadata:
        prep_obj.preprocessSiteMetadata()
    if relational_data:
        prep_obj.createRelationalData()