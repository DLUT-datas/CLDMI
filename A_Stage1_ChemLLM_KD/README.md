# KD
## About
This repository contains the code and resources of the Cross-modal Knowledge Distillation of the CLDMI:


## Overview of the framework
KeLGT is a novel self-supervised learning framework for the representation learning of molecular graphs, consisting of a novel graph transformer architecture, LiGhT, and a knowledge-guided HSSL pre-training strategy.


# 镜像环境 from AutoDL
    PyTorch  2.8.0
    Python  3.12(ubuntu22.04)
    CUDA  12.8 
    GPU RTX PRO 6000(96GB) 

## 搭建环境
    conda create -n llm_dkj python=3.10
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128或pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    pip install --pre dgl --no-deps -f https://data.dgl.ai/wheels-test/torch-2.4/cu124/repo.html或pip install dgl -f https://data.dgl.ai/wheels/cu128/repo.html



## Step 1: Prepare dataset 
The processed datasets, including molecular fingerprints, molecular descriptors, and auxiliary features, are available in the Zenodo repository (https://zenodo.org/records/21512952). Users can directly use these files for model training and evaluation.

For users who want to apply CLDMI to new datasets, the preprocessing pipeline is provided in this repository. The raw dataset only needs to be replaced with the target dataset following the same format, and the remaining preprocessing steps can be performed using the provided scripts.

    1. split PubChem324kV2_merged_dkj and chembl29 datasets with 95:5 by the balanced scaffold_split:
        cd datasets
        python data_scaffold_split.py  

    2. Extract the molecular descriptors of the SMILES in the PubChem324kV2_merged_dkj dataset:
        python preprocess_pretrain_dataset_stage1.py --data_path ./datasets/PubChem324kV2_merged_dkj_filtered/process_raw_scaffold/train  --dataset PubChem324kV2_merged_dkj_filtered_train
        python preprocess_pretrain_dataset_stage1.py --data_path ./datasets/PubChem324kV2_merged_dkj_filtered/process_raw_scaffold/valid  --dataset PubChem324kV2_merged_dkj_filtered_valid



## Step 2: Pre-train


    Step 1 Training of the dimensional adapter:
        python pretrain_chemllm_KD_step1.py --output_model_dir ./pretrained_models_dim_adapter --batch_size 32

    Step 2 Cross-modal knowledge distillation with prompt guidance : 
        python pretrain_chemllm_KD_step2.py --output_model_dir ./pretrained_models_PgKD_student --batch_size 32



