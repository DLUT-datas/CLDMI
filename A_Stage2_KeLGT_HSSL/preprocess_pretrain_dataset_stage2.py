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
import sys
sys.path.append("..")


from rdkit.Chem import rdMolDescriptors

from scripts.pubchemfp import GetPubChemFPs

from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler

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

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True) # './PubChemSTM_merged_dkj/process_raw_scaffold'  './chembl29/process_raw_scaffold'
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32) #32
    args = parser.parse_args()
    return args


def sklearn_standard_normalize(matrix):
    """
    使用scikit-learn进行标准化（推荐方法）
    """
    # Z-score标准化
    scaler = StandardScaler()
    normalized = scaler.fit_transform(matrix)
    
    return normalized, scaler


def preprocess_dataset_4_stage2(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}.csv")
    smiless = df.SMILES.values.tolist()

    # # ========================================================================================
    # print('extracting fingerprints')
    # FP_list = []
    # for smiles in smiless:
    #     mol = Chem.MolFromSmiles(smiles)
    #     FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    # FP_arr = np.array(FP_list)
    # FP_sp_mat = sp.csc_matrix(FP_arr)
    # print('saving fingerprints')
    # sp.save_npz(f"{args.data_path}/{args.dataset}/rdkfp1-7_512.npz", FP_sp_mat)
    # del FP_list, FP_arr, FP_sp_mat

    
    # ========================================================================================
    print('extracting PubChemFP')
    FP_list2 = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            FP_list2.append(list(GetPubChemFPs(mol)))

    FP_arr2 = np.array(FP_list2)
    FP_sp_mat2 = sp.csc_matrix(FP_arr2)
    print('saving PubChemFP fingerprints')
    sp.save_npz(f"{args.data_path}/{args.dataset}_pubchemfp.npz", FP_sp_mat2)
    del FP_list2, FP_arr2, FP_sp_mat2



    # ========================================================================================
    print('extracting MACCS FP')
    FP_list3 = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 计算分子的MACCS指纹
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            FP_list3.append(list(fp))

    FP_arr3 = np.array(FP_list3)
    FP_sp_mat3 = sp.csc_matrix(FP_arr3)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/{args.dataset}_maccsfp.npz", FP_sp_mat3)
    del FP_list3, FP_arr3, FP_sp_mat3



    # # ========================================================================================
    # print('extracting Morgan FP')
    # FP_list4 = []
    # for smiles in smiless:
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None:
    #         # 计算分子的Morgan指纹
    #         fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    #         FP_list4.append(fp_morgan)


    # FP_arr4 = np.array(FP_list4)
    # FP_sp_mat4 = sp.csc_matrix(FP_arr4)
    # print('saving fingerprints')
    # sp.save_npz(f"{args.data_path}/{args.dataset}/morganfp_1024.npz", FP_sp_mat4)
    # del FP_list4, FP_arr4, FP_sp_mat4




    # # ========================================================================================
    # print('extracting ErGf FP')
    # FP_list5 = []
    # for smiles in smiless:
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None:
    #         # 计算分子的ErGF指纹
    #         FP_list5.append(list(AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)))

    # FP_arr5 = np.array(FP_list5)
    # FP_sp_mat5 = sp.csc_matrix(FP_arr5)
    # print('saving fingerprints')
    # sp.save_npz(f""{args.data_path}/{args.dataset}/ergffp_441.npz", FP_sp_mat5)
    # del FP_list5, FP_arr5, FP_sp_mat5



    # ========================================================================================

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/{args.dataset}_mds.npz",md=arr[:,1:])




if __name__ == '__main__':
    args = parse_args()

    preprocess_dataset_4_stage2(args)


    # python preprocess_pretrain_dataset_stage2.py --data_path ./datasets/chembl29/process_raw_scaffold/train  --dataset chembl29_train
    # python preprocess_pretrain_dataset_stage2.py --data_path ./datasets/chembl29/process_raw_scaffold/valid  --dataset chembl29_valid 