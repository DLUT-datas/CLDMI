# CLDMI
## About
This repository contains the code and resources of the following paper:

ChemLLM-Distilled Multimodal Information Integration for Molecular Property Prediction
    

## Overview of the framework
CLDMI is a multimodal molecular representation learning framework designed for molecular property prediction. It combines ChemLLM knowledge distillation, knowledge-enhanced line graph Transformer learning, and feature-wise gated representation fusion to learn complementary molecular representations from multiple views.

By transferring chemical knowledge from large language models into lightweight molecular encoders and integrating structural information through multimodal fusion, CLDMI provides an efficient and effective solution for molecular property prediction without requiring text inputs during inference.


## Environment Setup

The experiments were conducted using Python 3.7, PyTorch 1.12.0, CUDA 11.6, and RDKit.

Create the conda environment:

```bash
    conda env create -f environment.yml
    conda activate CLDMI
```

### Stage 3 Fine-tuning with representation-level fusion and decision-level ensemble

### Step 1: Prepare dataset

The processed datasets, including molecular fingerprints, molecular descriptors, and auxiliary features, are available in the Zenodo repository (https://zenodo.org/records/21512952). Users can directly use these files for model training and evaluation.

For users who want to apply CLDMI to new datasets, the preprocessing pipeline is provided in this repository. The raw dataset only needs to be replaced with the target dataset following the same format, and the remaining preprocessing steps can be performed using the provided scripts.

    Construct molecular line graphs and extract the molecular descriptors and the fingerprints from SMILES in a downstream dataset (e.g., bace):

    python preprocess_downstream_dataset.py --data_path ./datasets/ --dataset bace 


## (e) Independent fine-tuning of pretrained encoders 

    1. student model branch: bash scripts_finetune/run_TSM_cmpn.sh

    2. KeLGT branch: bash scripts_finetune/run_kelgt_HSSL.sh

## (f) Representation-level fusion and decision-level ensemble

    bash scripts_finetune/run_fusion_CLDMI.sh


## Ablation of Knowledge Distillation and Hierarchical SSL (Q4)

# CMPN branch: 
    1. student model(CLDMI-G): 
        bash scripts_finetune/run_TSM_cmpn.sh

    2. student model variant removed the prompt mechanism during dimensional adapter training(CLDMI-G_T_wo_PT): 
        bash scripts_finetune/run_TSM_cmpn_T_wo_Pt_S_w_Pt.sh

    2. student model variant removed the prompt mechanism during student-side knowledge distillation(CLDMI-G_S_wo_PT): 
        bash scripts_finetune/run_TSM_cmpn_T_w_Pt_S_wo_Pt.sh

    4. student model variant w.o. knowledge distillation(CLDMI-G_wo_KD): 
        bash scripts_finetune/run_TSM_cmpn_wo_KD.sh

# KeLGT branch:
    1. KeLGT with Hierarchical SSL (CLDMI-KLG): 
        bash scripts_finetune/run_kelgt_HSSL.sh

    2. KeLGT variant removed the molecular-level SA prediction objective (CLDMI-KLG_wo_SA): 
        bash scripts_finetune/run_kelgt_HSSL_wo_SA.sh



## Ablation Study of Representation-level Fusion (Q5)
    
    1. two pretrained encoders joint fine-tuning in an end-to-end manner (CLDMI-JF): 
        bash scripts_finetune/run_fusion_joint_finetune_bs32.sh

    # decouples view-specific representation learning from cross-view integration

    2. decoupled feature-wise gated representation-level fusion (CLDMI-FGRF): 
        bash scripts_finetune/run_fusion_FGRF.sh

    3. decoupled concatenation fusion (CLDMI-CF): 
        bash scripts_finetune/run_fusion_ablation_concat.sh

    4. decoupled average fusion (CLDMI-AvgF): 
        bash scripts_finetune/run_fusion_ablation_avg.sh

    5. decoupled adaptive weighted fusion (CLDMI-AWF): 
        bash scripts_finetune/run_fusion_ablation_adaptive.sh


## Ablation of Decision-level Ensemble Integration (Q6)

    bash scripts_finetune/run_fusion_CLDMI.sh



## **Evaluation**

Here, we provide the fine-tuned model for 12 datasets in our Zenodo repository: https://zenodo.org/records/21512952, to guarantee the reproducibility of the test results reported in our paper.

Step 1: Download finetuned models:
    
    To download the fine-tuned models provied in the Zenodo repository (https://zenodo.org/records/21512952).
    Then unzip it and put the files in the CLDMI directory.

Step 2: Reproduce the results:

    1. Fine-tuning with representation-level fusion and decision-level ensemble: bash scripts_evaluate/run_evaluate_CLDMI.sh

    2. Independently fine-tune the CMPN branch: bash scripts_evaluate/run_evaluate_TSM_cmpn.sh

    3. Independently fine-tune the KeLGT branch: bash scripts_evaluate/run_evaluate_kelgt_HSSL.sh

    4. Independently fine-tune the FGRF module: bash scripts_evaluate/run_fusion_evaluate_FGRF.sh

    ……



<!-- ## Citation -->


