import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from random import Random
import torch
from torch.utils.data import Subset
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold


from rdkit.Chem import rdMolDescriptors

# from pubchemfp import GetPubChemFPs

from rdkit.Chem import AllChem


import argparse
import random


import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
import pandas as pd

import os
import random
import numpy as np
import torch
import dgl

from src.utils import set_random_seed


def balanced_scaffold_split_chembl29(data_df, frac=None, balanced=True, include_chirality=False, ramdom_state=0):

    print("data_df:", data_df[:5])
    data_cids = data_df['ID'].values
    data_smiless = data_df['SMILES'].values
    

    print(len(data_smiless))

    cid_list = []
    mol_list = []
    for cid, smile in zip(data_cids, data_smiless):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print('Invalid mol found')
            print('smile:', smile)
            continue
        else:
            cid_list.append(cid)
            mol_list.append(mol)

    # mol_list = []
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    # scaffolds_sets = defaultdict(list)
    scaffolds_sets = defaultdict(set)
    # for idx, mol in enumerate(mol_list):
    for cid, mol in zip(cid_list, mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        # scaffolds_sets[scaffold].append(idx)
        scaffolds_sets[scaffold].add(cid)

    random = Random(ramdom_state)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            # test_index += index_set
            if n_total_test > 0:
                test_index += index_set
            else:
                valid_index += index_set

    # print("train_index:", train_index)
    return train_index, valid_index, test_index
    # return  CustomSubset(dataset, train_index, type='train'), CustomSubset(dataset, valid_index), CustomSubset(dataset, test_index)


def balanced_scaffold_split_pubchemSTM(data_df, frac=None, balanced=True, include_chirality=False, ramdom_state=0):

    print("data_df:", data_df[:5])
    data_cids = data_df['CID'].values
    data_smiless = data_df['SMILES'].values
    

    print(len(data_smiless))

    cid_list = []
    mol_list = []
    for cid, smile in zip(data_cids, data_smiless):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print('Invalid mol found')
            print('smile:', smile)
            continue
        else:
            cid_list.append(cid)
            mol_list.append(mol)

    # mol_list = []
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    # scaffolds_sets = defaultdict(list)
    scaffolds_sets = defaultdict(set)
    # for idx, mol in enumerate(mol_list):
    for cid, mol in zip(cid_list, mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        # scaffolds_sets[scaffold].append(idx)
        scaffolds_sets[scaffold].add(cid)

    random = Random(ramdom_state)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            # test_index += index_set
            if n_total_test > 0:
                test_index += index_set
            else:
                valid_index += index_set

    print("train_index:", train_index, len(train_index))
    print("valid_index:", valid_index, len(valid_index))
    # print("test_index:", test_index, len(test_index))
    return train_index, valid_index, test_index
    # return  CustomSubset(dataset, train_index, type='train'), CustomSubset(dataset, valid_index), CustomSubset(dataset, test_index)


# 使用示例
if __name__ == "__main__":

    set_random_seed(2025)

    input_csv = "./PubChem324kV2_merged_dkj/PubChem324kV2_merged_dkj.csv"  # 替换为你的输入文件路径

    data_df = pd.read_csv(input_csv)
    data_smiless = data_df['SMILES'].values

    train_index, valid_index, _ = balanced_scaffold_split_pubchemSTM(data_df, frac=[0.95, 0.05, 0], balanced=True, include_chirality=False, ramdom_state=2025)
    print("data_smiless:", data_smiless[:5])
    print("valid_index:", valid_index[:5])

    # train_df = data_df[data_df['CID'].isin(train_index)].set_index('CID')
    # valid_df = data_df[data_df['CID'].isin(valid_index)].set_index('CID')

    train_df = (data_df[data_df['CID'].isin(train_index)]
            .set_index('CID')
            .reindex(train_index)
            .reset_index())
    
    valid_df = (data_df[data_df['CID'].isin(valid_index)]
            .set_index('CID')
            .reindex(valid_index)
            .reset_index())

    os.makedirs("./PubChem324kV2_merged_dkj/process_raw_scaffold/train", exist_ok=True)
    os.makedirs("./PubChem324kV2_merged_dkj/process_raw_scaffold/valid", exist_ok=True)

    train_df.to_csv("./PubChem324kV2_merged_dkj/process_raw_scaffold/train/PubChem324kV2_merged_dkj_train.csv", index=False)
    valid_df.to_csv("./PubChem324kV2_merged_dkj/process_raw_scaffold/valid/PubChem324kV2_merged_dkj_valid.csv", index=False)

    print("valid_df:", valid_df[:5])


    # # # -----------------------------------------------------------chembl29------------------------------------------------------------------
    # input_csv = "./chembl29/chembl29.csv"  # 替换为你的输入文件路径

    # data_df = pd.read_csv(input_csv)

    # # 添加序号索引列（从0开始）
    # df_with_index = data_df.reset_index()
    # df_with_index.columns = ['ID', 'SMILES']  # 重命名列
    # print("df_with_index:", df_with_index[:5])
    # # df_with_index.to_csv("./chembl29/chembl29_with_index.csv", index=False)

    # # data_smiless = data_df['SMILES'].values

    # train_index, valid_index, _ = balanced_scaffold_split_chembl29(df_with_index, frac=[0.95, 0.05, 0], balanced=True, include_chirality=False, ramdom_state=2025)
    # print("valid_index:", valid_index[:5])


    # train_df = (df_with_index[df_with_index['ID'].isin(train_index)]
    #         .set_index('ID')
    #         .reindex(train_index)
    #         .reset_index())
    
    # valid_df = (df_with_index[df_with_index['ID'].isin(valid_index)]
    #         .set_index('ID')
    #         .reindex(valid_index)
    #         .reset_index())


    # os.makedirs("./chembl29/process_raw_scaffold/train", exist_ok=True)
    # os.makedirs("./chembl29/process_raw_scaffold/valid", exist_ok=True)

    # train_df.to_csv("./chembl29/process_raw_scaffold/train/chembl29_train.csv", index=False)
    # valid_df.to_csv("./chembl29/process_raw_scaffold/valid/chembl29_valid.csv", index=False)

    # print("valid_df:", valid_df[:5])
    # # -----------------------------------------------------------chembl29------------------------------------------------------------------


