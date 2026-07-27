import sys
sys.path.append('..')

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
from chemprop.parsing_4_TSM_pretrain import add_train_args, modify_train_args
from chemprop.nn_utils import initialize_weights

from src.data.pretrain_dataset import PubChem_MoleculeDataset_4_stage2
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict
from src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from src.data.collator import Collator_pretrain, Collator_pretrain_stage2
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from src.trainer.scheduler import PolynomialDecayLR, get_polynomial_decay_schedule_with_warmup
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



# last: 使用最后一个token，适合生成式模型
# mean: 使用平均值，更稳定但可能稀释重要信息
# cls: 使用第一个token，如果有特殊标记
def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # 计算每个序列非padding token的平均值
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # 获取每个序列最后一个非padding token的位置
    sequence_lengths = attention_mask.sum(dim=1) - 1  # 减1因为索引从0开始
    batch_size = hidden_states.shape[0]
    
    # 收集最后一个token的隐藏状态
    last_token_indices = sequence_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
    last_token_embeddings = hidden_states.gather(1, last_token_indices).squeeze(1)
    
    return last_token_embeddings


def _cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    # 使用第一个token作为表示
    return hidden_states[:, 0, :]


def train_student_distill_form_Teacher(
    epoch,
    train_dataloader,
    val_dataloader,
    text_model, text_tokenizer,
    molecule_graph_model,  
    scaler):  # 添加scaler参数

    max_length = 512
    reg_loss_fn = MSELoss(reduction='none')
    results = []
    results_dir = './pretrain_loss_stages2'
    os.makedirs(results_dir, exist_ok=True) 


    # 在epoch开始时清零梯度
    optimizer.zero_grad()

    text_model.eval()
    molecule_graph_model.train()

    for step, batch in enumerate(train_dataloader):
        single_result = []
        description_batch, smiles_batch, linegraph_batch, fps_batch, mds_batch, prompt_mds_batch = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        linegraph_batch, fps_batch, mds_batch, prompt_mds_batch = linegraph_batch.to(device), fps_batch.to(device), mds_batch.to(device), prompt_mds_batch.to(device)



        # 使用autocast包装前向传播
        with autocast(enabled=args.use_amp):

            with torch.no_grad():   
                # ChemLLM
                inputs = text_tokenizer(description_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length, return_attention_mask=True).to(device)
                outputs = text_model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]  # [B, L, H]
                description_repr = mean_pooling(last_hidden_states, inputs['attention_mask'])
                latent_description_repr = text_model.dim_adapter_encoder(description_repr) 
                # molecule_text_predictive_md = text_model.prompt_predictor(latent_description_repr)
                # reconstructed_description_repr = text_model.dim_adapter_decoder(latent_description_repr) 
            
            # molecule_graph_repr
            molecule_graph_repr, molecule_graph_predictive_md = molecule_graph_model(smiles_batch, None)

            # 重构损失
            mse_loss = F.mse_loss(molecule_graph_repr, latent_description_repr)

            # 预测损失
            predictive_loss = reg_loss_fn(molecule_graph_predictive_md, prompt_mds_batch).mean() 

            # 批次内损失，广播机制
            diff_molecule_graph_predictive_md = molecule_graph_predictive_md.unsqueeze(1) - molecule_graph_predictive_md.unsqueeze(0)
            diff_true_md = prompt_mds_batch.unsqueeze(1) - prompt_mds_batch.unsqueeze(0)
            batch_diff_loss = reg_loss_fn(diff_molecule_graph_predictive_md, diff_true_md).mean() 

            train_batch_loss = mse_loss + predictive_loss + batch_diff_loss
        

        
        # 使用scaler进行反向传播
        scaler.scale(train_batch_loss).backward()

        # 使用scaler进行梯度裁剪（如果需要）
        if args.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(molecule_graph_model.parameters(), args.max_grad_norm)
            
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()


        # 定期清理CUDA缓存
        if step % 1000 == 0:
            torch.cuda.empty_cache()

        # 验证部分
        if step == len(train_dataloader)-1:
            
            text_model.eval()
            molecule_graph_model.eval()

            # 验证预训练模型  不计算梯度，不更新参数
            val_total_loss, val_total_mse_loss, val_total_predictive_loss, val_total_batch_diff_loss = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for val_step, batch in enumerate(val_dataloader):
                    single_result = []
                    description_batch, smiles_batch, linegraph_batch, fps_batch, mds_batch, prompt_mds_batch = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

                    linegraph_batch, fps_batch, mds_batch, prompt_mds_batch = linegraph_batch.to(device), fps_batch.to(device), mds_batch.to(device), prompt_mds_batch.to(device)

                    # 验证时也使用autocast以确保一致性
                    with autocast(enabled=args.use_amp):

                        # ChemLLM
                        inputs = text_tokenizer(description_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length, return_attention_mask=True).to(device)
                        outputs = text_model(**inputs, output_hidden_states=True)
                        last_hidden_states = outputs.hidden_states[-1]  # [B, L, H]
                        description_repr = mean_pooling(last_hidden_states, inputs['attention_mask'])
                        latent_description_repr = text_model.dim_adapter_encoder(description_repr) 

                        # molecule_graph_repr
                        molecule_graph_repr, graph_prompt_preds = molecule_graph_model(smiles_batch, None)

                        # mse
                        val_mse_loss = F.mse_loss(molecule_graph_repr, latent_description_repr)

                        # 预测损失
                        val_predictive_loss = reg_loss_fn(graph_prompt_preds, prompt_mds_batch).mean() 

                        # 批次内损失，广播机制
                        diff_molecule_graph_predictive_md = graph_prompt_preds.unsqueeze(1) - graph_prompt_preds.unsqueeze(0)
                        diff_true_md = prompt_mds_batch.unsqueeze(1) - prompt_mds_batch.unsqueeze(0)
                        val_batch_diff_loss = reg_loss_fn(diff_molecule_graph_predictive_md, diff_true_md).mean() 

                        val_loss = val_mse_loss + val_predictive_loss  + val_batch_diff_loss 
                        val_total_loss += val_loss.item()
                        val_total_mse_loss += val_mse_loss.item()
                        val_total_predictive_loss += val_predictive_loss.item()
                        val_total_batch_diff_loss += val_batch_diff_loss.item()

            #-----------------------------------------------
            single_result.append(['epochs', epoch, 'val_total_loss', "%.6f" % val_total_loss, 'val_mse_loss', "%.6f" % val_total_mse_loss, 'val_predictive_loss', "%.6f" % val_total_predictive_loss, 'val_batch_diff_loss', "%.6f" % val_total_batch_diff_loss])
            df = pd.DataFrame(single_result)
            df.to_csv(
                results_dir + '/{}.csv'.format('TSM_val_loss_202512091'),
                mode='a', index=False, header=False
            )
            #-----------------------------------------------

            save_model(save_best=False, epoch=epoch, stage=2)


        # 获取当前日期
        current_date = datetime.date.today()
        current_time = time.strftime("%H:%M:%S")
        combined = f"{current_date} {current_time}"
        
        # 记录损失值
        results.append(['epochs', epoch, 'date', combined, 'train_step', step, 'train_loss', "%.6f" % train_batch_loss.detach().item(), 'mse_loss', "%.6f" % mse_loss.detach().item(), 'predictive_loss', "%.6f" % predictive_loss.detach().item(), 'batch_diff_loss', "%.6f" % batch_diff_loss.detach().item()])
        
        if step % 100 == 0 or step == len(train_dataloader)-1:
            df = pd.DataFrame(results)
            df.to_csv(
                results_dir + '/{}.csv'.format('TSM_train_loss_202512091'),
                mode='a', index=False, header=False
            )
            results = []
    
    return


def save_model(save_best, epoch=None, stage=2):

    if args.output_model_dir is not None:
        os.makedirs(args.output_model_dir, exist_ok=True)

        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            model_file = "model_best.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)


        if stage == 1:
            # text, 保存文本适配器
            text_state_dict = {
                'dim_adapter_encoder': chemllm_model.dim_adapter_encoder.state_dict(),
                'dim_adapter_decoder': chemllm_model.dim_adapter_decoder.state_dict(),
                'text_predictor': chemllm_model.prompt_predictor.state_dict()
            }
            torch.save(text_state_dict, f"{args.output_model_dir}/text_{model_file}")

        elif stage == 2:
            # graph
            saved_file_path = os.path.join(args.output_model_dir, "molecule_graph_{}".format(model_file))
            torch.save(molecule_graph_model.state_dict(), saved_file_path)

    return


def create_dim_adapter_encoder(
    input_dim: int = 4096,
    intermediate_dims: list[int] = [2048, 1024, 512],
    latent_dim: int = 300,
    use_layer_norm: bool = True,
    dropout: float = 0.1,
    activation_fn: nn.Module = nn.GELU
    ) -> nn.Sequential:
    """
    创建维度适配器编码器
    """
    
    encoder_layers = []
    current_dim = input_dim
    
    # 构建中间编码层
    for hidden_dim in intermediate_dims:
        # 创建当前层
        layer_layers = [nn.Linear(current_dim, hidden_dim)]
        
        # 层归一化（可选）
        if use_layer_norm:
            layer_layers.append(nn.LayerNorm(hidden_dim))
        
        # 激活函数和Dropout
        layer_layers.append(activation_fn())
        if dropout > 0:
            layer_layers.append(nn.Dropout(dropout))
        
        encoder_layers.append(nn.Sequential(*layer_layers))
        current_dim = hidden_dim
    
    # 构建最终层（到潜在空间）
    # 最后一层通常不加激活函数，以保持潜在空间的灵活性
    final_layer_layers = [nn.Linear(current_dim, latent_dim)]
    if use_layer_norm:
        final_layer_layers.append(nn.LayerNorm(latent_dim))
    
    encoder_layers.append(nn.Sequential(*final_layer_layers))
    
    return nn.Sequential(*encoder_layers)


def create_dim_adapter_decoder(
    latent_dim: int = 300,
    intermediate_dims: list[int] = [512, 1024, 2048],
    output_dim: int = 4096,
    use_layer_norm: bool = True,
    dropout: float = 0.1,
    activation_fn: nn.Module = nn.GELU
    ) -> nn.Sequential:
    """
    创建维度适配器解码器
    """
    
    decoder_layers = []
    current_dim = latent_dim
    
    # 构建中间解码层
    for i, hidden_dim in enumerate(intermediate_dims):
        # 创建当前层
        layer_layers = [nn.Linear(current_dim, hidden_dim)]
        
        # 非最终层添加归一化和激活
        if i < len(intermediate_dims) - 1:  # 不是最后一层
            if use_layer_norm:
                layer_layers.append(nn.LayerNorm(hidden_dim))
            layer_layers.append(activation_fn())
            if dropout > 0:
                layer_layers.append(nn.Dropout(dropout))
        else:  # 最后一层（中间层的最后一层）
            if use_layer_norm:
                layer_layers.append(nn.LayerNorm(hidden_dim))
            layer_layers.append(activation_fn())
            # 最后一层中间层不加dropout
        
        decoder_layers.append(nn.Sequential(*layer_layers))
        current_dim = hidden_dim
    
    # 构建最终层（回到输入维度）
    # 最终层通常不加激活，因为重建的是原始特征（可能有负值）
    final_layer_layers = [nn.Linear(current_dim, output_dim)]
    decoder_layers.append(nn.Sequential(*final_layer_layers))
    
    return nn.Sequential(*decoder_layers)


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

    
    modify_train_args(args)
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

    set_random_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    kwargs = {}

    # dkj
    if args.dataset == "PubChem324kV2": 
        dataset_root = os.path.join(args.dataspace_path, "PubChem324kV2_merged_dkj/process_raw_scaffold")
        train_dataset = PubChem_MoleculeDataset_4_stage2(dataset_root, type='train')
        val_dataset = PubChem_MoleculeDataset_4_stage2(dataset_root, type='valid')
    else:
        raise Exception

    # Teacher 
    if args.text_type == "ChemLLM":

        # 本地模型和tokenizer的路径
        local_directory = "./ChemLLM-7B-Chat-1_5-SFT_2"
        adapter_checkpoint_path = "./pretrained_models_TSM_20251208/text_model_20.pth"

        chemllm_model = AutoModelForCausalLM.from_pretrained(local_directory, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        chemllm_tokenizer = AutoTokenizer.from_pretrained(local_directory, trust_remote_code=True)

        # ============== dim_adapter ==============  
        chemllm_model.dim_adapter_encoder = create_dim_adapter_encoder(
                                            input_dim=4096,
                                            intermediate_dims=[2048, 1024, 512],
                                            latent_dim=300,
                                            use_layer_norm=True,
                                            dropout=0.1
                                        ).to(device)
        
        # 加载检查点
        checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
        
        # 加载编码器
        if 'dim_adapter_encoder' in checkpoint:
            chemllm_model.dim_adapter_encoder.load_state_dict(
                checkpoint['dim_adapter_encoder']
            )
            print("✓ dim_adapter_encoder参数加载成功")

        kwargs["text_tokenizer"] = chemllm_tokenizer
        kwargs["text_model"] = chemllm_model

       
    # Student 
    if args.molecule_MPNN == "CMPNN":

        molecule_graph_model = build_pretrain_model(args).cuda()
        args.graph_model_path = "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"
        if args.graph_model_path != None:
            molecule_graph_model.encoder.load_state_dict(torch.load(args.graph_model_path), strict=False)
            print("torch.load_graph_model, CMPN加载成功!")
        else:
            print("随机初始化graph_model!")

        kwargs["molecule_graph_model"] = molecule_graph_model  # 300


    collator = Collator_pretrain_stage2(vocab, max_length=config['path_length'], n_virtual_nodes=2)

    train_loader = torch_DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, collate_fn=collator)
    val_loader = torch_DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, collate_fn=collator)


    # 创建梯度缩放器
    scaler = GradScaler(enabled=args.use_amp)
    args.init_lr_pretrain = 1e-5
    
    model_param_group = [{"params": molecule_graph_model.parameters(), "lr": args.init_lr_pretrain}]

    optimizer = optim.AdamW(model_param_group, weight_decay=args.weight_decay) # weight_decay
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=3000, tot_updates=30000, lr=args.init_lr_pretrain, end_lr=1e-8, power=1)
    # lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.n_epochs * len(train_dataset) // args.batch_size // 10, args.n_epochs * len(train_dataset) // args.batch_size, 1e-7)
    # lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.n_epochs * len(train_dataset) // (args.batch_size * args.accumulation_steps) // 10, args.n_epochs * len(train_dataset) // (args.batch_size * args.accumulation_steps), 1e-5)
    
    
    kwargs["scaler"] = scaler  # 将scaler添加到kwargs中
    print("kwargs:", kwargs)

        
    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))

        for param in chemllm_model.parameters():
            param.requires_grad = False
        for param in molecule_graph_model.ffn.parameters():
            param.requires_grad = True

        train_student_distill_form_Teacher(e, train_loader, val_loader, **kwargs)


      
# python pretrain_chemllm_TSM_step2.py --output_model_dir ./pretrained_models_TSM_20251209 --batch_size 32