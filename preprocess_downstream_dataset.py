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

from rdkit.Chem import rdMolDescriptors
from scripts.pubchemfp import GetPubChemFPs
from rdkit.Chem import AllChem
import random
from random import Random
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args

def preprocess_dataset(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.dataset}_{args.path_length}.pkl"
    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                                   smiless,
                                   max_length=args.path_length,
                                   n_virtual_nodes=2,
                                   n_jobs=args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('saving graphs')
    save_graphs(cache_file_path, valid_graphs,
                labels={'labels': labels})

    # print('extracting fingerprints')
    # FP_list = []
    # for smiles in smiless:
    #     mol = Chem.MolFromSmiles(smiles)
    #     FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    # FP_arr = np.array(FP_list)
    # FP_sp_mat = sp.csc_matrix(FP_arr)
    # print('saving fingerprints')
    # sp.save_npz(f"{args.data_path}/{args.dataset}/rdkfp1-7_512.npz", FP_sp_mat)


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
    sp.save_npz(f"{args.data_path}/{args.dataset}/pubchemfp_881.npz", FP_sp_mat2)
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
    sp.save_npz(f"{args.data_path}/{args.dataset}/maccsfp_167.npz", FP_sp_mat3)
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
    # sp.save_npz(f"{args.data_path}/{args.dataset}/ergffp_441.npz", FP_sp_mat5)
    # del FP_list5, FP_arr5, FP_sp_mat5



    # ========================================================================================

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/{args.dataset}/molecular_descriptors.npz",md=arr[:,1:])

if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)

    # python preprocess_downstream_dataset.py --data_path ./datasets/ --dataset bace 