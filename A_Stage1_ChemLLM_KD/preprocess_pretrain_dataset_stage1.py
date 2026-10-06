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


def preprocess_dataset_4_stage1(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}.csv")
    smiless = df.SMILES.values.tolist()


    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/{args.dataset}_mds.npz",md=arr[:,1:])


    mds_200_name = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
                    'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
                    'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                    'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
                    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
                    'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                    'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex',
                    'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
                    'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
                    'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount',
                    'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
                    'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                    'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
                    'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                    'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount',
                    'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
                    'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
                    'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                    'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                    'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
                    'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                    'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                    'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
                    'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
                    'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                    'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                    'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
                    'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                    'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                    'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                    'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                    'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                    'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                    'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                    'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
                    'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
                    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                    'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']

    print("arr:", arr.shape)
    md_df = pd.DataFrame(data=arr[:,1:], columns=mds_200_name)
    md_df.to_csv(f"{args.data_path}/{args.dataset}_mds.csv")


    selected_md = md_df.loc[:, ['TPSA', 'MolLogP', 'BertzCT']]
    selected_md.to_csv(f"{args.data_path}/{args.dataset}_selected_mds.csv")


    # data = np.load(md_path)['md'].astype(np.float32)
    # Z-score标准化
    normalized_selected_md, scaler = sklearn_standard_normalize(selected_md.values)
    print("Z-score标准化结果:")
    print("normalized_selected_md:", normalized_selected_md.shape)
    print(normalized_selected_md[:, 0])

    np.savez_compressed(f"{args.data_path}/{args.dataset}_normalized_selected_mds.npz", md=normalized_selected_md)


    print("selected_md.values:", selected_md.values.shape)


if __name__ == '__main__':
    args = parse_args()

    preprocess_dataset_4_stage1(args)


