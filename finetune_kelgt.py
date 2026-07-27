import sys
sys.path.append('..')

from src.utils import set_random_seed, balanced_scaffold_split
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from torch.nn import MSELoss, BCEWithLogitsLoss, SmoothL1Loss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset, MoleculeDataset_KeLG
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.trainer.finetune_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
import warnings
warnings.filterwarnings("ignore")

import os, csv
import pandas as pd

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_threads", type=int, default=8)
    
    # Experiment settings
    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=["classification", 'regression'])
    parser.add_argument("--metric", type=str, required=True, choices=['roc-auc', 'ap', 'acc', 'rmse', 'mae', 'r2', 'spearman', 'pearson'])
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--predict_drp", type=float, default=0)
    parser.add_argument("--save_smiles_splits", type=str, default=True)
    
    # Training hyperparameters
    parser.add_argument("--lgt_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_norm_label", action='store_true')

    # Model hyperparameters
    parser.add_argument("--n_predictor_layers", type=int, default=2)
    parser.add_argument("--d_predictor_hidden", type=int, default=256)

    parser.add_argument("--eval_kelgt", action='store_true', default=False, help='Kelgt.eval()')
    


    args = parser.parse_args()
    return args

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

def seed_worker(worker_id):
    """
    Set random seed
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

def finetune(args):
    """
    Fine-tune model
    """
    set_random_seed(args.seed)
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])
    print("device:", device)

    # new add
    data_df = pd.read_csv(os.path.join(args.data_path, f"{args.dataset}/{args.dataset}.csv"))
    smiless = data_df['smiles'].tolist()
    samples_num = len(smiless)
    args.task_names = data_df.columns.drop(['smiles']).tolist()

    if args.dataset_type == 'classification':
        config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0
        # config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = args.predict_drp

    elif args.dataset_type == 'regression':
        if samples_num < 200:
            config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0.5
        elif samples_num < 500:
            config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0.4   
        elif samples_num < 1000:
            config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0.3
        elif samples_num < 2000:
            config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0.2
        elif samples_num < 5000:
            config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0.1
        else:
            config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = 0.0
    
    # config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = args.predict_drp
    # config['attn_drop'] = config['feat_drop'] = config['predict_drop'] = args.predict_drp = 0.2
    
    args.lgt_config = args.config
    args.result_save_path=args.save_path + '/' + str(args.dataset)+'_'+str(config['predict_drop'])
    args.model_save_path=args.save_path + '/' + str(args.dataset)+'_'+str(config['predict_drop']) + '/' + str(args.seed)
    # args.result_save_path=args.save_path + '/' + str(args.dataset)
    # args.model_save_path=args.save_path + '/' + str(args.dataset) + '/' + str(args.seed)

    args.kelgt_model_save_path='./result_finetuned_kelgt_HSSL/' + str(args.dataset) + '_'+ str(config['attn_drop']) + '/' + str(args.seed)
    # args.g_model_save_path='./result_TSM_finetuned_CMPN/' + str(args.dataset) + '/hold_out_' + str(args.seed) + '/model_0'
    print("args.kelgt_model_save_path:", args.kelgt_model_save_path)
    
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)


    train_index, val_index, test_index = balanced_scaffold_split(data=smiless, frac=None, balanced=True, include_chirality=False, ramdom_state=args.seed)
    print("split on seed:", args.split)
    print(len(train_index), len(val_index), len(test_index))
    
    if args.save_smiles_splits:
        # with open(args.data_path, 'r') as f:
        with open(os.path.join(args.data_path, f"{args.dataset}/{args.dataset}.csv"), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        # all_split_indices = []
        for indexs, name in [(train_index, 'train'), (val_index, 'val'), (test_index, 'test')]: # index
            # with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
            with open(os.path.join(args.model_save_path, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                # for smiles in dataset.smiles():
                for index in indexs:
                    writer.writerow([smiless[index]])
            # with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
            with open(os.path.join(args.model_save_path, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # for smiles in dataset.smiles():
                #     writer.writerow(lines_by_smiles[smiles])
                for index in indexs:
                    writer.writerow(lines_by_smiles[smiless[index]])
        #     split_indices = []
        #     for smiles in dataset.smiles():
        #         split_indices.append(indices_by_smiles[smiles])
        #         split_indices = sorted(split_indices)
        #     all_split_indices.append(split_indices)

        # with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
        #     pickle.dump(all_split_indices, f)


    # Dataset loading
    # train_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='train')
    # val_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='val')
    # test_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='test')
    train_dataset = MoleculeDataset_KeLG(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, index=train_index)
    val_dataset = MoleculeDataset_KeLG(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, index=val_index)
    test_dataset = MoleculeDataset_KeLG(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, index=test_index)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    # Model loading
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
        input_drop=0,
        attn_drop=config['attn_drop'],
        feat_drop=config['feat_drop'],
        n_node_types=vocab.vocab_size
    ).to(device)
    
    # Fine-tuning settings
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()})
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()}, strict=False)
    # model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=train_dataset.n_tasks, n_layers=args.n_predictor_layers, predictor_drop=args.dropout, device=device, d_hidden_feats=args.d_predictor_hidden)
    # 预测器使用dropout
    model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=train_dataset.n_tasks, n_layers=args.n_predictor_layers, predictor_drop=config['predict_drop'], device=device, d_hidden_feats=args.d_predictor_hidden)
    
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    print("Total model parameters: {:.2f}M".format(sum(x.numel() for x in model.parameters()) / 1e6))

    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, args.lgt_epochs * len(train_dataset) // args.batch_size // 10, args.lgt_epochs * len(train_dataset) // args.batch_size, 1e-9)

    if args.metric in ['spearman', 'pearson']:
        loss_fn = SmoothL1Loss(reduction='none')
    elif args.metric in ['mae', 'rmse']:
        loss_fn = MSELoss(reduction='none')
    else:
        loss_fn = BCEWithLogitsLoss(reduction='none')

    if args.dataset_type == 'regression' and (not args.no_norm_label):
        mean, std = train_dataset.mean.numpy(), train_dataset.std.numpy()
    else:
        mean, std = None, None

    if args.dataset_type == 'classification':
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks)
    else:
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks, mean=mean, std=std)

    result_tracker = Result_Tracker(args.metric)
    summary_writer = None

    # Fine-tuning
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, label_std=torch.from_numpy(std).to(device) if std is not None else None)
    best_train, best_val, best_test = trainer.fit(model, train_loader, val_loader, test_loader)
    print(f"Training: {best_train:.3f}, Validation: {best_val:.3f}, Test: {best_test:.3f}")

    return best_train, best_val, best_test

if __name__ == '__main__':

    # args = parse_args()
    # finetune(args)


    set_random_seed(2025)
    args = parse_args()
    config = config_dict[args.config]
    best_train, best_val, best_test = finetune(args)
    print(f"val: {best_val:.6f}, test: {best_test:.6f}")
    result_single={'sacffold_seed':[args.seed], 'drp': [config['predict_drop']], 'lr': [args.lr], 'weight_decay': [args.weight_decay], 'val': [best_val], 'test': [best_test]}
    df = pd.DataFrame(result_single)

    if os.path.isfile(args.result_save_path + '/result_KeLGT.csv'):
        df.to_csv(args.result_save_path + '/result_KeLGT.csv', mode='a', header=False, index=False)
    else:
        df.to_csv(args.result_save_path + '/result_KeLGT.csv', mode='a', index=False)

