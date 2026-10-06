import sys
from contextlib import nullcontext
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import dgl
import numpy as np
import os
import random
from src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from src.data.pretrain_dataset import MoleculeDataset, MoleculeDataset_pubchem_maccs, MoleculeDataset_Pretrain_w_SA, MoleculeDataset_Pretrain_only_SA
from src.data.collator import Collator_pretrain, Collator_pretrain_only_SA
# from src.model.light import LiGhTPredictor as LiGhT
from src.model.light_w_SA_prompt import LiGhTPredictor as LiGhT
from src.trainer.scheduler import PolynomialDecayLR
# from src.trainer.pretrain_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict

import pandas as pd
import time, datetime
import torch
import numpy as np
from sklearn.metrics import f1_score
import pdb
import torch.nn.functional as F
from torch import nn


import warnings
warnings.filterwarnings("ignore")
local_rank = int(os.environ['LOCAL_RANK'])

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def init_params(module):
    """
    Initialize model parameters
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    """
    Get predictor
    """
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, reg_evaluator, clf_evaluator, result_tracker, summary_writer, device, ddp=False, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.sl_loss_fn = sl_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator = clf_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0
        
        # 移除混合精度训练相关的代码
        # self.scaler = torch.cuda.amp.GradScaler()
        
        # 梯度累积步数
        self.accumulation_steps = args.accumulation_steps
        
    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sa_scores) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        # sl_labels = sl_labels.to(self.device)
        # disturbed_fps = disturbed_fps.to(self.device)
        # disturbed_mds = disturbed_mds.to(self.device)
        # print("disturbed_mds:", type(disturbed_mds))
        sa_scores = sa_scores.to(self.device)
        
        # 移除混合精度前向传播
        # sl_predictions, fp_predictions, md_predictions, sa_predictions = model(batched_graph, disturbed_fps, disturbed_mds)
        sa_predictions = model(batched_graph, fps, mds)
        # mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask']>=1].cpu().numpy()
            
        # return mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, sa_scores, sa_predictions
        return sa_scores, sa_predictions
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        results = []

        results_dir = '../pretrain_loss_only_SA_grad_accum'
        os.makedirs(results_dir, exist_ok=True)

        self.optimizer.zero_grad(set_to_none=True)

        accumulation_loss = 0.0
        accumulation_sa_loss = 0.0

        # 只使用完整的梯度累计组，保证每次 optimizer update 的
        # effective batch size 完全一致。
        num_batches = len(train_loader)
        usable_batches = (
            num_batches // self.accumulation_steps
        ) * self.accumulation_steps

        for batch_idx, batched_data in enumerate(train_loader):
            if batch_idx >= usable_batches:
                break

            is_update_step = (
                (batch_idx + 1) % self.accumulation_steps == 0
            )

            # DDP 下，非最后一个 micro-batch 不同步梯度。
            sync_context = (
                model.no_sync()
                if self.ddp and not is_update_step
                else nullcontext()
            )

            with sync_context:
                sa_scores, sa_predictions = self._forward_epoch(
                    model, batched_data
                )

                sa_loss = self.reg_loss_fn(
                    sa_predictions, sa_scores
                ).mean()
                loss = sa_loss

                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()

            # 仅保存数值，避免累计带 grad_fn 的 Tensor。
            accumulation_loss += loss.detach().item()
            accumulation_sa_loss += sa_loss.detach().item()

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5
                )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # scheduler 仍按 optimizer update 计步。
                self.lr_scheduler.step()
                self.n_updates += 1

                avg_loss = (
                    accumulation_loss / self.accumulation_steps
                )
                avg_sa_loss = (
                    accumulation_sa_loss / self.accumulation_steps
                )

                current_date = datetime.date.today()
                current_time = time.strftime("%H:%M:%S")
                combined = f"{current_date} {current_time}"

                results.append([
                    'epochs', epoch_idx,
                    'update_step', self.n_updates,
                    'date', combined,
                    'train_loss', avg_loss,
                    'sa_loss', avg_sa_loss,
                ])

                accumulation_loss = 0.0
                accumulation_sa_loss = 0.0

                if self.n_updates % 50 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(
                        results_dir + '/{}_{}.csv'.format(
                            'pretrain_train_loss', 'only_SA_260910'
                        ),
                        mode='a',
                        index=False,
                        header=False
                    )
                    results = []

                if self.n_updates >= self.args.n_steps:
                    break

        if len(results) > 0:
            df = pd.DataFrame(results)
            df.to_csv(
                results_dir + '/{}_{}.csv'.format(
                    'pretrain_train_loss', 'only_SA_260910'
                ),
                mode='a',
                index=False,
                header=False
            )

    def validation_epoch(self, model, validation_loader, epoch_idx):
        model.eval()
        val_epoch_total_loss = 0.0
        # sl_total_loss = 0.0
        # fp_total_loss = 0.0
        # md_total_loss = 0.0
        sa_total_loss = 0.0
        val_single_result = []
        results_dir = '../pretrain_loss_only_SA_grad_accum'
        os.makedirs(results_dir, exist_ok=True) 

        with torch.no_grad():
            for batch_idx, batched_data in enumerate(validation_loader):
                # 验证时移除混合精度
                sa_scores, sa_predictions = self._forward_epoch(model, batched_data)              
                # sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
                # fp_loss = self.clf_loss_fn(fp_predictions, fps).mean() #* 0.1
                # md_loss = self.reg_loss_fn(md_predictions, mds).mean()

                sa_loss = self.reg_loss_fn(sa_predictions, sa_scores).mean()
                loss = sa_loss


                val_epoch_total_loss += loss.item()
                # sl_total_loss += sl_loss.item()
                # fp_total_loss += fp_loss.item()
                # md_total_loss += md_loss.item()
                sa_total_loss += sa_loss.item()


            # ===== ✅ 统一在 epoch 结束后求均值 =====
            num_batches = len(validation_loader)

            val_loss = val_epoch_total_loss / num_batches
            # sl_loss = sl_total_loss / num_batches
            # fp_loss = fp_total_loss / num_batches
            # md_loss = md_total_loss / num_batches
            sa_loss = sa_total_loss / num_batches

            self.save_model(model, epoch_idx)
            
            val_single_result.append(['epochs', epoch_idx, 'val_loss', "%.16f" % val_loss, 
                                    # 'sl_loss', "%.16f" % sl_loss, 
                                    # 'fp_loss', "%.16f" % fp_loss, 
                                    # 'md_loss', "%.16f" % md_loss,
                                    'sa_loss', "%.16f" % sa_loss,
                                    ])

            df = pd.DataFrame(val_single_result)
            df.to_csv(
                results_dir + '/{}_{}.csv'.format('pretrain_val_loss', 'only_SA_260910'),
                mode='a', index=False, header=False
            )

    def fit(self, model, train_loader, validation_loader):  # 添加validation_loader参数
        for epoch in range(0, 1001): # 1001
            print("epoch:", epoch)

            # if epoch == 0:
            #     model.eval()
            #     self.validation_epoch(model, validation_loader, epoch)

            epoch+=1
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.n_updates >= self.args.n_steps:
                model.eval()
                self.validation_epoch(model, validation_loader, epoch)
                break
            model.eval()
            self.validation_epoch(model, validation_loader, epoch)

    def save_model(self, model, epoch=None):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        if epoch is None:
            torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}.pth")
        else:
            torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}_{epoch}.pth")

if __name__ == '__main__':
    args = parse_args()
    config = config_dict[args.config]
    print(config)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    print(local_rank)
    val_results, test_results, train_results = [], [], []
    
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    # collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
    collator = Collator_pretrain_only_SA(vocab, max_length=config['path_length'], n_virtual_nodes=2)
    
    # train_dataset = MoleculeDataset(root_path=args.data_path, type='train')
    # train_dataset = MoleculeDataset_pubchem_maccs(root_path=args.data_path, type='train')
    # train_dataset = MoleculeDataset_Pretrain_w_SA(root_path=args.data_path, type='train')
    train_dataset = MoleculeDataset_Pretrain_only_SA(
        root_path=args.data_path, type='train' # valid 
    )

    # config['batch_size'] 作为希望保持不变的全局 effective batch size。
    # 例如 batch_size=256, world_size=1, accumulation_steps=4：
    # 每次 forward 的 micro-batch=64，但有效 batch size 仍为 256。
    world_size = torch.distributed.get_world_size()
    effective_batch_size = config['batch_size']
    batch_divisor = world_size * args.accumulation_steps

    if effective_batch_size % batch_divisor != 0:
        raise ValueError(
            f"config['batch_size']={effective_batch_size} 必须能被 "
            f"world_size({world_size}) * "
            f"accumulation_steps({args.accumulation_steps}) "
            f"= {batch_divisor} 整除。"
        )

    micro_batch_size = effective_batch_size // batch_divisor

    if micro_batch_size < 1:
        raise ValueError("micro_batch_size 必须 >= 1")

    if local_rank == 0:
        print(
            f"Gradient accumulation: "
            f"effective_batch_size={effective_batch_size}, "
            f"micro_batch_size_per_gpu={micro_batch_size}, "
            f"world_size={world_size}, "
            f"accumulation_steps={args.accumulation_steps}"
        )

    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=micro_batch_size,
        num_workers=args.n_threads,
        worker_init_fn=seed_worker,
        drop_last=True,
        collate_fn=collator
    )

    validation_dataset = MoleculeDataset_Pretrain_only_SA(
        root_path=args.data_path, type='valid'
    )

    # 验证无 backward，先保留原验证 batch size。
    # 如果验证也 OOM，可以单独减小，不影响训练 effective batch size。
    validation_batch_size = effective_batch_size // world_size

    validation_loader = DataLoader(
        validation_dataset,
        sampler=DistributedSampler(
            validation_dataset, shuffle=False
        ),
        batch_size=validation_batch_size,
        num_workers=args.n_threads,
        worker_init_fn=seed_worker,
        drop_last=False,
        collate_fn=collator
    )

    print("len_train_loader:", len(train_loader))
    print("len_validation_loader:", len(validation_loader))
    
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=config['input_drop'],
        attn_drop=config['attn_drop'],
        feat_drop=config['feat_drop'],
        n_node_types=vocab.vocab_size
    ).to(device)
    # print("Total model parameters: {:.2f}M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=1, n_layers=2, predictor_drop=config['feat_drop'], device=device, d_hidden_feats=256)
    print("Total model parameters: {:.2f}M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=20000, tot_updates=200000,lr=config['lr'], end_lr=1e-9,power=1)
    reg_loss_fn = MSELoss(reduction='none')
    clf_loss_fn = BCEWithLogitsLoss(reduction='none')
    sl_loss_fn = CrossEntropyLoss(reduction='none')

    reg_metric, clf_metric = "r2", "rocauc_resp"
    reg_evaluator = Evaluator("chembl29", reg_metric, train_dataset.d_mds)
    clf_evaluator = Evaluator("chembl29", clf_metric, train_dataset.d_fps)
    result_tracker = Result_Tracker(reg_metric)
    if local_rank == 0:
        summary_writer = SummaryWriter(f"tensorboard/pretrain-{args.config}", )
    else: 
        summary_writer = None
    trainer = Trainer(args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, reg_evaluator, clf_evaluator, result_tracker, summary_writer, device=device, ddp=True, local_rank=local_rank)
    trainer.fit(model, train_loader, validation_loader)
    if local_rank == 0:
        summary_writer.close()
    torch.distributed.destroy_process_group()
    
    


    
    
# unset OMP_NUM_THREADS
# CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port 12312 pretrain_kelgt_only_SA_grad_accum.py --save_path ../models_only_SA_260910/pretrained/base_dkj --n_threads 8 --n_devices 1 --config base  --data_path ../datasets/chembl29/process_raw_scaffold/ --n_steps 200000