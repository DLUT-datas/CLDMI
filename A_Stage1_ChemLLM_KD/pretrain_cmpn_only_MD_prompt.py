# import sys
# sys.path.append('..')

import os
import time, datetime
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader

from argparse import Namespace
# from MoleculeSTM.PubChemSTM import PubChemSTM_324k_pretrain_Datasets, PubChemSTM_324k_pretrain_val_Datasets

import random
import pandas as pd
from argparse import Namespace
from chemprop.models import MoleculeModel
# from chemprop.parsing import add_train_args, modify_train_args
from chemprop.parsing_4_KD_pretrain import add_train_args, modify_train_args
from chemprop.nn_utils import initialize_weights

from src.data.pretrain_dataset import PubChem_TextMoleculeDataset
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict
from src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from src.data.collator import Collator_pretrain, Collator_pretrain_KD
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from src.trainer.scheduler import PolynomialDecayLR#, get_polynomial_decay_schedule_with_warmup
import pdb
import gc

# 导入混合精度训练所需的库
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils import set_random_seed


def build_pretrain_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    args.ffn_hidden_size = args.hidden_size//2
    args.output_size = args.hidden_size

    model = MoleculeModel(classification=True, multiclass=False, pretrain=True)
    model.create_encoder(args)
    model.create_ffn(args)
    
    initialize_weights(model)

    return model



def train_student_w_Prompt(epoch, train_dataloader, val_dataloader, molecule_graph_model, scaler):  # 添加scaler参数

    max_length = 512
    batch_results = []
    results_dir = './pretrain_loss_variant_only_MDprompt'
    os.makedirs(results_dir, exist_ok=True) 


    molecule_graph_model.train()
    train_total_loss, train_total_distill_loss, train_total_predictive_loss, train_total_batch_diff_loss = 0.0, 0.0, 0.0, 0.0

    for step, batch in enumerate(train_dataloader):

        description_batch, smiles_batch, prompt_mds_batch = batch[0], batch[1], batch[2]
        prompt_mds_batch = prompt_mds_batch.to(device)

        # 每个batch开始前清空梯度
        optimizer.zero_grad()

        # 使用autocast包装前向传播
        with autocast(enabled=args.use_amp):
            
            # molecule_graph_model
            molecule_graph_repr, molecule_graph_predictive_md = molecule_graph_model(smiles_batch, None)

            
            # 预测损失
            predictive_loss = F.mse_loss(molecule_graph_predictive_md, prompt_mds_batch, reduction='mean')

            # 批次内损失，广播机制
            diff_molecule_graph_predictive_md = molecule_graph_predictive_md.unsqueeze(1) - molecule_graph_predictive_md.unsqueeze(0)
            diff_true_md = prompt_mds_batch.unsqueeze(1) - prompt_mds_batch.unsqueeze(0)
            batch_diff_loss = F.mse_loss(diff_molecule_graph_predictive_md, diff_true_md, reduction='mean')

            train_batch_loss = predictive_loss + batch_diff_loss
         
        # 使用scaler进行反向传播
        scaler.scale(train_batch_loss).backward()

        if args.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(molecule_graph_model.parameters(), args.max_grad_norm)
            
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        train_total_loss += train_batch_loss.item()
        train_total_predictive_loss += predictive_loss.item()
        train_total_batch_diff_loss +=  batch_diff_loss.item()
        
        # 获取当前日期
        current_date = datetime.date.today()
        current_time = time.strftime("%H:%M:%S")
        combined = f"{current_date} {current_time}"
        
        # 记录损失值
        batch_results.append(['epochs', epoch, 'date', combined, 'train_step', step, 'train_loss', "%.6f" % train_batch_loss.item(), \
            'predictive_loss', "%.6f" % predictive_loss.item(), 'batch_diff_loss', "%.6f" % batch_diff_loss.item()])
        
        if step % 100 == 0 or step == len(train_dataloader)-1:
            df = pd.DataFrame(batch_results)
            df.to_csv(
                results_dir + '/{}.csv'.format('MDprompt_train_batch_loss'),
                mode='a', index=False, header=False
            )
            batch_results = []
    

    # Epoch-level Training Loss
    num_train_batches = len(train_dataloader)

    if num_train_batches == 0:
        raise RuntimeError("train_dataloader is empty.")

    train_total_loss /= num_train_batches
    train_total_predictive_loss /= num_train_batches
    train_total_batch_diff_loss /= num_train_batches

    train_epoch_result = [['epochs', epoch, 'date', combined, 'train_total_loss', "%.6f" % train_total_loss, \
        'train_predictive_loss', "%.6f" % train_total_predictive_loss, 'train_batch_diff_loss', "%.6f" % train_total_batch_diff_loss]]
    
    df = pd.DataFrame(train_epoch_result)
    df.to_csv(os.path.join(results_dir, 'MD_prompt_train_loss.csv'),
        mode='a', index=False, header=False)

    # 验证部分
    # 验证预训练模型  不计算梯度，不更新参数        
    molecule_graph_model.eval()
    val_total_loss, val_total_distill_loss, val_total_predictive_loss, val_total_batch_diff_loss = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):

            description_batch, smiles_batch, prompt_mds_batch = batch[0], batch[1], batch[2]

            prompt_mds_batch = prompt_mds_batch.to(device)

            # 验证时也使用autocast以确保一致性
            with autocast(enabled=args.use_amp):

                # molecule_graph_model
                molecule_graph_repr, graph_prompt_preds = molecule_graph_model(smiles_batch, None)

                # 预测损失
                val_predictive_loss = F.mse_loss(graph_prompt_preds, prompt_mds_batch, reduction='mean')

                # 批次内损失，广播机制
                diff_molecule_graph_predictive_md = graph_prompt_preds.unsqueeze(1) - graph_prompt_preds.unsqueeze(0)
                diff_true_md = prompt_mds_batch.unsqueeze(1) - prompt_mds_batch.unsqueeze(0)
                val_batch_diff_loss = F.mse_loss(diff_molecule_graph_predictive_md, diff_true_md, reduction='mean') 

                val_loss = val_predictive_loss  + val_batch_diff_loss 
            val_total_loss += val_loss.item()
            val_total_predictive_loss += val_predictive_loss.item()
            val_total_batch_diff_loss += val_batch_diff_loss.item()

    # Average Validation Loss
    num_val_batches = len(val_dataloader)

    if num_val_batches == 0:
        raise RuntimeError(
            "val_dataloader is empty."
        )


    val_total_loss /= num_val_batches
    val_total_predictive_loss /= num_val_batches
    val_total_batch_diff_loss /= num_val_batches

    #-----------------------------------------------
    val_epoch_result = [['epochs', epoch, 'val_total_loss', "%.6f" % val_total_loss, 'val_predictive_loss', "%.6f" % val_total_predictive_loss, 'val_batch_diff_loss', "%.6f" % val_total_batch_diff_loss]]
    df = pd.DataFrame(val_epoch_result)
    df.to_csv(
        results_dir + '/{}.csv'.format('MD_prompt_val_loss'),
        mode='a', index=False, header=False
    )
    #-----------------------------------------------

    save_model(save_best=False, epoch=epoch)

    return val_total_loss


def save_model(save_best, epoch=None):

    if args.output_model_dir is not None:
        os.makedirs(args.output_model_dir, exist_ok=True)

        if save_best:
            model_file = "model_best.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        # graph
        saved_file_path = os.path.join(args.output_model_dir, "molecule_graph_{}".format(model_file))
        torch.save(molecule_graph_model.state_dict(), saved_file_path)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    print("arguments\t", args)


    args.use_input_features = False
    args.ffn_output_size = 3
    args.n_epochs = 20
    args.use_amp = False
    args.max_grad_norm = 5.0
    args.output_size = args.ffn_output_size
    
    modify_train_args(args)
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

    set_random_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    kwargs = {}

    # dkj
    if args.dataset == "PubChem324kV2_filtered": 
        dataset_root = os.path.join(args.dataspace_path, "PubChem324kV2_merged_dkj_filtered/process_raw_scaffold")
        train_dataset = PubChem_TextMoleculeDataset(dataset_root, type='train') # train
        val_dataset = PubChem_TextMoleculeDataset(dataset_root, type='valid')
    else:
        raise Exception



    
    # Student 
    if args.molecule_MPNN == "CMPNN":

        molecule_graph_model = build_pretrain_model(args).cuda()
        print("随机初始化graph_model!")
        kwargs["molecule_graph_model"] = molecule_graph_model  # 300


    collator = Collator_pretrain_KD(vocab, max_length=config['path_length'], n_virtual_nodes=2)

    train_loader = torch_DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, collate_fn=collator)
    val_loader = torch_DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, collate_fn=collator)


    # 创建梯度缩放器
    scaler = GradScaler(enabled=args.use_amp)
    args.init_lr_pretrain = 1e-5
    
    model_param_group = [{"params": molecule_graph_model.parameters(), "lr": args.init_lr_pretrain}]

    optimizer = optim.AdamW(model_param_group, weight_decay=args.weight_decay) # weight_decay
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=2000, tot_updates=20000, lr=args.init_lr_pretrain, end_lr=1e-7, power=1)
    # lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=2000, tot_updates=20000, lr=args.init_lr_pretrain, end_lr=1e-8, power=1)
    
    kwargs["scaler"] = scaler  # 将scaler添加到kwargs中
    print("kwargs:", kwargs)
  

    best_val_loss = float("inf")
    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))

        val_loss = train_student_w_Prompt(e, train_loader, val_loader, **kwargs)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_model(save_best=True, epoch=e)

        print(f"Best model updated: "f"epoch={e}, "f"val_loss={best_val_loss:.6f}")

      
# python pretrain_cmpn_only_MD_prompt.py --output_model_dir ./pretrained_CMPN_from_only_MDprompt --batch_size 32