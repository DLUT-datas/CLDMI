# KeLGT
## About
This repository contains the code and resources of the HSSL of the KeLGT:


## Overview of the framework
KeLGT is a novel self-supervised learning framework for the representation learning of molecular graphs, consisting of a novel graph transformer architecture, LiGhT, and a knowledge-guided HSSL pre-training strategy.


## Step 1: Prepare dataset 
The processed datasets, including molecular fingerprints, molecular descriptors, and auxiliary features, are available in the Zenodo repository (https://zenodo.org/records/21512952). Users can directly use these files for model training and evaluation.

For users who want to apply CLDMI to new datasets, the preprocessing pipeline is provided in this repository. The raw dataset only needs to be replaced with the target dataset following the same format, and the remaining preprocessing steps can be performed using the provided scripts.

    1. split PubChem324kV2_merged_dkj and chembl29 datasets with 95:5 by the balanced scaffold_split:
        cd datasets
        python data_scaffold_split.py  

    2. Extract the molecular descriptors and fingerprints of the SMILES in the ChEMBL dataset:
        python preprocess_pretrain_dataset_stage2.py --data_path ./datasets/chembl29/process_raw_scaffold/train  --dataset chembl29_train
        python preprocess_pretrain_dataset_stage2.py --data_path ./datasets/chembl29/process_raw_scaffold/valid  --dataset chembl29_valid

    3. Compute molecular SA scores for chembl29 dataset by the SA.ipynb file;


## Step 2: Pre-train
### Stage 2 Knowledge-guided hierarchical SSL
# 镜像环境 from AutoDL
    PyTorch  2.0.0
    Python  3.8(ubuntu20.04)
    CUDA  11.8
    GPU RTX 4090(24GB)


    Step 1 Training of the dimensional adapter:
        python pretrain_chemllm_TSM_step1.py --output_model_dir ./pretrained_models_TSM_20251208 --batch_size 32

    Step 2 Prompt-guided cross-modal knowledge distillation: 
        python pretrain_chemllm_TSM_step2.py --output_model_dir ./pretrained_models_TSM_20251209 --batch_size 32



## **Setup environment**

Setup the required environment using `environment.yml` with Anaconda. While in the project directory run:

    conda env create

Activate the environment

    conda activate KPGT
