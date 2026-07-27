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
from src.fusion_model.fusion_model import FeatureGatedResidualFusion, MultimodalModel
from src.trainer.finetune_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
import warnings
warnings.filterwarnings("ignore")

import os, csv
import pandas as pd


# from rdkit import RDLogger  
# RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
from typing import Tuple


from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs, get_loss_func, get_metric_func
from chemprop.parsing import modify_train_args, update_checkpoint_args
from chemprop.utils import create_logger
from chemprop.features import get_available_features_generators
import random
import pandas as pd
import time
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Molecular Property Prediction")
    # 公共参数
    add_common_args(parser)

    # CMPNN参数
    add_cmpnn_args(parser)

    # LineGraph Transformer参数
    add_lgt_args(parser)

    # Fusion参数
    add_fusion_args(parser)

    args = parser.parse_args()
    return args

def add_common_args(parser):
    # --------------------
    # experiment
    # --------------------
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_type", choices=["classification", "regression"])
    parser.add_argument("--metric", type=str, required=True, choices=['roc-auc', 'ap', 'acc', 'rmse', 'mae', 'r2', 'spearman', 'pearson'])
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--save_smiles_splits", action="store_true")
    parser.add_argument("--n_threads", type=int,default=8)
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='Directory where model checkpoints will be saved')
    parser.add_argument("--dropout", type=float, default=0)
    
def add_lgt_args(parser):

    parser.add_argument("--lgt_config", default="base")
    parser.add_argument("--lgt_lr", type=float, default=3e-5)
    parser.add_argument("--lgt_batch_size", type=int, default=32)
    parser.add_argument("--lgt_epochs", type=int, default=50)
    parser.add_argument("--lgt_weight_decay", type=float, default=0)
    parser.add_argument("--lgt_predict_drop", type=float, default=0)
    parser.add_argument("--lgt_hidden", type=int, default=256)
    parser.add_argument("--lgt_predictor_layers", type=int, default=2)
    parser.add_argument("--lgt_model_path", type=str, required=True)
    parser.add_argument("--lgt_predictor_hidden", type=int, default=256)

    parser.add_argument("--eval_kelgt", action='store_true', default=False, help='Kelgt.eval()')

def add_cmpnn_args(parser):

    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    # parser.add_argument('--gpu', type=int, choices=list(range(torch.cuda.device_count())), help='Which GPU to use')
    # parser.add_argument('--data_path', type=str, help='Path to data CSV file', default='M_CYP1A2I_I.csv')
    parser.add_argument('--use_compound_names', action='store_true', default=False, help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int, help='Maximum number of data points to load')
    parser.add_argument('--test', action='store_true', default=False, help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False, help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*', choices=get_available_features_generators(), help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',help='Path to features to use in FNN (instead of features_generator)')                   
    # parser.add_argument('--save_dir', type=str, default='./ckpt', help='Directory where model checkpoints will be saved')
    # parser.add_argument('--save_smiles_splits', action='store_true', default=False, help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory from which to load model checkpoints' '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint (.pt file)')
    # parser.add_argument('--dataset_type', type=str,choices=['classification', 'regression', 'multiclass'], help='Type of dataset, e.g. classification or regression.'
    #                          'This determines the loss function used during training.',default='regression') # classification
    parser.add_argument('--multiclass_num_classes', type=int, default=3, help='Number of classes when running multiclass classification')
    parser.add_argument('--separate_val_path', type=str, help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str, help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*', help='Path to file with features for separate test set')
    # parser.add_argument('--split_type', type=str, default='random', choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'], help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1, help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str, help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str, help='Indices of files to use as train/val/test''Overrides --num_folds and --seed.')
    # parser.add_argument('--seed', type=int, default=0,help='Random seed to use when splitting data into train/val/test sets.'
    #                          'When `num_folds` > 1, the first fold uses this seed and all'
    #                          'subsequent folds add 1 to the seed.')
    # parser.add_argument('--metric', type=str, default=None,choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'], help='Metric to use during evaluation.'
    #                          'Note: Does NOT affect loss function used during training'
    #                          '(loss is determined by the `dataset_type` argument).'
    #                          'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False, help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False, help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')
    parser.add_argument('--no_features_scaling', action='store_true', default=False, help='Turn off scaling of features')
    
    # Training arguments
    parser.add_argument("--cmpnn_batch_size", type=int, default=64)
    parser.add_argument("--cmpnn_epochs", type=int, default=30)
    parser.add_argument("--cmpnn_init_lr", type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument("--cmpnn_max_lr", type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument("--cmpnn_final_lr", type=float, default=1e-4, help='Final learning rate')
    parser.add_argument('--cmpnn_warmup_epochs', type=float, default=2.0, help='Number of epochs during which learning rate increases linearly from'
                            'init_lr to max_lr. Afterwards, learning rate decreases exponentially' 'from max_lr to final_lr.')
    
    # parser.add_argument('--cmpnn_epochs', type=int, default=30, help='Number of epochs to run')
    # parser.add_argument('--cmpnn_batch_size', type=int, default=50, help='Batch size')
    
    # parser.add_argument('--init_lr', type=float, default=1e-4)
    # parser.add_argument('--max_lr', type=float, default=1e-3,help='Maximum learning rate')
    # parser.add_argument('--final_lr', type=float, default=1e-4, help='Final learning rate')
    # parser.add_argument('--no_features_scaling', action='store_true', default=False, help='Turn off scaling of features')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1, help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300, help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False, help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3, help='Number of message passing steps')
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'], help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False, help='Undirected edges (always sum the two relevant bond vectors)')                     
    parser.add_argument('--ffn_hidden_size', type=int, default=None, help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False, help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--aug_strategies', type=int, default=0)

    parser.add_argument('--load_from_KEGGCL', action='store_true', default=False, help='load_from_KEGGCL') 
    parser.add_argument('--load_from_TSM', action='store_true', default=False, help='load_from_TSM') 

    parser.add_argument('--eval_cmpn', action='store_true', default=False, help='cmpn.eval()') 

    parser.add_argument('--run_cmpn', action='store_true', default=False, help='run CMPN')
    parser.add_argument('--run_cmpn_branch', action='store_true', default=False, help='run CMPN branch')

def add_fusion_args(parser):

    parser.add_argument("--fusion_lr", type=float, default=5e-4)
    parser.add_argument("--fusion_weight_decay", type=float, default=0)
    parser.add_argument("--fusion_batch_size", type=int, default=32)
    parser.add_argument("--fusion_epochs", type=int, default=30)
    parser.add_argument("--fusion_hidden", type=int, default=256)
    parser.add_argument("--joint_finetune", action="store_true")

    parser.add_argument("--eval_fm", action="store_true", default=False, help='fusion_module.eval()')



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

def run_lgt(args):
    """
    Fine-tune model
    """
    set_random_seed(args.seed)
    config = config_dict[args.lgt_config]
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
    args.num_tasks = len(args.task_names)
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
    

    args.result_save_path=args.save_path + '/' + str(args.dataset)
    args.model_save_path=args.save_dir + '/' + str(args.dataset) + '/KeLGT_scaffold_split_' + str(args.seed)
    # args.result_save_path=args.save_path + '/' + str(args.dataset)
    # args.model_save_path=args.save_path + '/' + str(args.dataset) + '/' + str(args.seed)

    # args.kelgt_model_save_path='./results_finetune_kelgt_concate_fusion_dkj_Pret_0130_SBCE_only_w_sa_b256_e20/' + str(args.dataset) + '_'+ str(config['attn_drop']) + '/' + str(args.seed)

    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)


    train_index, val_index, test_index = balanced_scaffold_split(data=smiless, frac=None, balanced=True, include_chirality=False, ramdom_state=args.seed)
    print("split on seed:", args.split_type)
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


    # Dataset loading
    # train_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='train')
    # val_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='val')
    # test_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='test')
    train_dataset = MoleculeDataset_KeLG(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, index=train_index)
    val_dataset = MoleculeDataset_KeLG(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, index=val_index)
    test_dataset = MoleculeDataset_KeLG(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, index=test_index)
    train_loader = DataLoader(train_dataset, batch_size=args.lgt_batch_size, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.lgt_batch_size, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.lgt_batch_size, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    # val_loader_64 = DataLoader(val_dataset, batch_size=args.lgt_batch_size*2, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    # test_loader_64 = DataLoader(test_dataset, batch_size=args.lgt_batch_size*2, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

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

    # 预训练初始化文件
    # args.lgt_model_path ='./models_SBCE_only_w_SA_prompt_0130/pretrained/base_dkj/base_20.pth' 
    # kelgt加载
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()})
    # model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=train_dataset.n_tasks, n_layers=args.n_predictor_layers, predictor_drop=args.dropout, device=device, d_hidden_feats=args.d_predictor_hidden)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.lgt_model_path}').items()}, strict=False)
    model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=train_dataset.n_tasks, n_layers=args.lgt_predictor_layers, predictor_drop=config['predict_drop'], device=device, d_hidden_feats=args.lgt_predictor_hidden)
    
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    print("Total model parameters: {:.2f}M".format(sum(x.numel() for x in model.parameters()) / 1e6))

    optimizer = transformers.AdamW(model.parameters(), lr=args.lgt_lr, weight_decay=args.lgt_weight_decay)
    lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, args.lgt_epochs * len(train_dataset) // args.lgt_batch_size // 10, args.lgt_epochs * len(train_dataset) // args.lgt_batch_size, 1e-9)

    if args.metric in ['spearman', 'pearson']:
        loss_fn = SmoothL1Loss(reduction='none')
    elif args.metric in ['mae', 'rmse']:
        loss_fn = MSELoss(reduction='none')
    else:
        loss_fn = BCEWithLogitsLoss(reduction='none')

    if args.dataset_type == 'regression':# and (not args.no_norm_label):
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
    # best_train, best_val, best_test = trainer.fit_f(model, train_loader, val_loader, test_loader)
    val_preds_lgt, val_lgt, test_lgt, test_preds_lgt, test_labels_lgt = trainer.fit_f(model, train_loader, val_loader, test_loader)
    # print(f"Validation: {val_lgt:.3f}, Test: {test_lgt:.3f}")

    return val_preds_lgt, test_preds_lgt, val_lgt, test_lgt, evaluator, result_tracker, train_loader, val_loader, test_loader, model, mean, std, train_index, val_index, test_index

def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_cmpn(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """hold-out"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    makedirs(args.save_dir)
    save_dir = os.path.join(args.save_dir, args.dataset)
    makedirs(save_dir)
    task_names = get_task_names(args.data_path)
    print("task_names:", task_names)
    
    # makedirs(os.path.join(save_dir, args.dataset))
    # Run training on different random seeds for each hold-out
    all_scores = []
    args.num_runs = 1
    for run_num in range(args.num_runs): # 一个种子运行一次
 
        args.save_dir = os.path.join(save_dir, f'CMPN_sacffold_split_{init_seed}')
        makedirs(args.save_dir)
        model_scores, val_preds, test_preds, model_cmpn, train_data, val_data, test_data, train_index, val_index, test_index = run_training(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)


    # Report scores 
    for fold_num, scores in enumerate(all_scores):
        info(f'run_{fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'run_ {init_seed} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'result test on seed {args.seed}')
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    args.save_dir = save_dir
    return mean_score, val_preds, test_preds, model_cmpn, train_data, val_data, test_data, train_index, val_index, test_index

def set_fusion_only_train_mode(mmm):
    mmm.graph_module.eval()
    mmm.linegraph_module.eval()
    mmm.fusion_module.train()

    for p in mmm.graph_module.parameters():
        p.requires_grad = False
    for p in mmm.linegraph_module.parameters():
        p.requires_grad = False
    for p in mmm.fusion_module.parameters():
        p.requires_grad = True

def get_fusion_loss(args):
    if args.dataset_type == "classification":
        return nn.BCEWithLogitsLoss(reduction="none")
    elif args.dataset_type == "regression":
        return nn.MSELoss(reduction="none")
    else:
        raise ValueError(args.dataset_type)
    
@torch.no_grad()
def evaluate_fusion(
    mmm,
    data_loader,
    args,
    evaluator,
    device,
    label_mean,
    label_std
):
    mmm.eval()
    
    preds_fm_all, preds_cmpn_all, preds_lgt_all, labels_all = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            smiles, g, ecfp, md, labels = batch

            g = g.to(device)
            ecfp = ecfp.to(device)
            md = md.to(device)

            pred_fusion, _, pred_cmpn, pred_lgt = mmm(
                smiles=smiles,
                g=g,
                ecfp=ecfp,
                md=md
            )

            preds_fm_all.append(pred_fusion.detach().cpu())
            preds_cmpn_all.append(pred_cmpn.detach().cpu())
            preds_lgt_all.append(pred_lgt.detach().cpu())
            labels_all.append(labels.detach().cpu())

    preds_fm_all = torch.cat(preds_fm_all, dim=0)
    preds_cmpn_all = torch.cat(preds_cmpn_all, dim=0)
    preds_lgt_all = torch.cat(preds_lgt_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    # print("preds_all:", type(preds_all))
    # print("labels_all:", type(labels_all))

    score = evaluator.eval(labels_all, preds_fm_all)

    if args.dataset_type == 'regression':
        preds_fm_all = preds_fm_all * label_std.detach().cpu() + label_mean.detach().cpu()

    return score, preds_fm_all, preds_cmpn_all, preds_lgt_all, labels_all

def train_fusion_stage(args, model_cmpn, model_lgt, fusion_module, train_loader, val_loader, test_loader, evaluator, device, label_mean, label_std):
    
    mmm = MultimodalModel(
        graph_module=model_cmpn,
        linegraph_module=model_lgt,
        fusion_module=fusion_module
    ).to(device)

    set_fusion_only_train_mode(mmm)

    loss_func = get_fusion_loss(args)

    optimizer_fm = torch.optim.AdamW(
        mmm.parameters(), # 只优化融合模型的参数
        lr=args.fusion_lr,
        weight_decay=args.fusion_weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_fm,
        T_max=args.fusion_epochs,
        eta_min=1e-6
    )

    best_val = -float("inf") if args.dataset_type == "classification" else float("inf")
    best_epoch = 0

    args.f_model_save_path=args.save_dir + '/FM_scaffold_split_' + str(args.seed)
    os.makedirs(args.f_model_save_path, exist_ok=True)
    save_path = os.path.join(args.f_model_save_path, "fusion_model.pth") 


    if args.eval_fm == False:
        
        for epoch in range(1, args.fusion_epochs + 1):

            # 每个 epoch 开始都重新设置：
            # CMPN eval + freeze; LGT eval + freeze; Fusion train + trainable
            set_fusion_only_train_mode(mmm)
            # print("cmpn training:", mmm.graph_module.training)
            # print("lgt training:", mmm.linegraph_module.training)
            # print("fusion training:", mmm.fusion_module.training)  
            # print("cmpn grad:", next(mmm.graph_module.parameters()).requires_grad)
            # print("lgt grad:", next(mmm.linegraph_module.parameters()).requires_grad)
            # print("fusion grad:", next(mmm.fusion_module.parameters()).requires_grad)

            loss_sum = 0.0
            n_batches = 0

            for batch in train_loader:
                smiles, g, ecfp, md, labels = batch

                g = g.to(device)
                ecfp = ecfp.to(device)
                md = md.to(device)
                labels = labels.to(device) # 回归任务是原始标签

                mask = ~torch.isnan(labels)
                labels = torch.nan_to_num(labels, nan=0.0)

                if (label_mean is not None) and (label_std is not None):
                    labels = (labels - label_mean)/label_std

                optimizer_fm.zero_grad()

                pred_fusion, fusion_info, _, _ = mmm(
                    smiles=smiles,
                    g=g,
                    ecfp=ecfp,
                    md=md
                )

                loss_fm = loss_func(pred_fusion, labels) * mask.float()
                loss_fm = loss_fm.sum() / mask.sum().clamp(min=1)
                loss_fm.backward()
                torch.nn.utils.clip_grad_norm_(mmm.fusion_module.parameters(), max_norm=5.0)
                optimizer_fm.step()

                loss_sum += loss_fm.item()
                n_batches += 1

            train_loss = loss_sum / max(n_batches, 1)

        
            val_score, val_preds_fm, val_preds_cmpn, val_preds_lgt, val_labels = evaluate_fusion(
                mmm=mmm,
                data_loader=val_loader,
                args=args,
                evaluator=evaluator,
                device=device,
                label_mean=label_mean, 
                label_std=label_std
            )

            scheduler.step()

            improved = (
                val_score > best_val
                if args.dataset_type == "classification"
                else val_score < best_val
            )

            print(f"Epoch {epoch:03d} | "f"loss={train_loss:.6f} | "f"val={val_score:.6f}")
            if improved:
                best_val = val_score
                best_epoch = epoch
                torch.save(mmm.fusion_module.state_dict(), save_path)
                print("update best fusion:", epoch)

        mmm.fusion_module.load_state_dict(torch.load(save_path, map_location=device))

    else:
        mmm.fusion_module.load_state_dict(torch.load(save_path, map_location=device))

        
    val_score, val_preds_fm, val_preds_cmpn, val_preds_lgt, val_labels = evaluate_fusion(
        mmm=mmm,
        data_loader=val_loader,
        args=args,
        evaluator=evaluator,
        device=device,
        label_mean=label_mean, 
        label_std=label_std
    )

    test_score, test_preds_fm, test_preds_cmpn, test_preds_lgt, test_labels = evaluate_fusion(
        mmm=mmm,
        data_loader=test_loader,
        args=args,
        evaluator=evaluator,
        device=device,
        label_mean=label_mean, 
        label_std=label_std
    )

    print(f"Best epoch={best_epoch}, "f"best val={best_val:.6f}, "f"test={test_score:.6f}")

    # return mmm, best_val, test_score, test_preds, test_labels
    # return val_preds, test_preds, best_val, test_score, val_labels, test_labels, mmm
    return val_preds_fm, val_preds_cmpn, val_preds_lgt, test_preds_fm, val_preds_cmpn, val_preds_lgt, best_val, test_score, val_labels, test_labels, mmm

def ensemble_three_prediction(

        test_preds_cmpn,
        test_preds_lgt,
        test_preds_fm,
        test_labels,
        evaluator,
        dataset_type,
        label_mean, 
        label_std
    ):

    avg_test_pred = (np.asarray(test_preds_cmpn) + np.asarray(test_preds_lgt) + np.asarray(test_preds_fm))/3.0

    if dataset_type == 'regression':
        avg_test_pred = (avg_test_pred - label_mean) / label_std

    avg_test_score = evaluator.eval(np.asarray(test_labels), avg_test_pred)

    print("=" * 60)
    print("Avg. weight, Test score:", avg_test_score)

    return avg_test_score, avg_test_pred


# def search_best_weight_three(
#         pred1,
#         pred2,
#         pred3,
#         labels,
#         evaluator,
#         dataset_type,
#         label_mean,
#         label_std,
#         step=0.05):

#     pred1 = np.asarray(pred1)
#     pred2 = np.asarray(pred2)
#     pred3 = np.asarray(pred3)
#     labels = np.asarray(labels)
#     # print("pred1:", type(pred1))
#     # print("pred2:", type(pred2))
#     # print("pred3:", type(pred3))
#     # print("labels:", type(labels))

#     if dataset_type == "classification":
#         best_score = -1
#     else:
#         best_score = 1e10

#     best_weight = None
#     grid = np.arange(0, 1 + 1e-8, step)
#     for w1 in grid:
#         for w2 in grid:
#             w3 = 1 - w1 - w2
#             if w3 < 0:
#                 continue

#             weighted_pred = w1 * pred1 + w2 * pred2 + w3 * pred3

#             if dataset_type == 'regression':
#                 weighted_pred = (weighted_pred - label_mean) / label_std
#             score = evaluator.eval(labels, weighted_pred)

#             if dataset_type == "classification":
#                 if score > best_score:
#                     best_score = score
#                     best_weight = (w1, w2, w3)
#             else:
#                 if score < best_score:
#                     best_score = score
#                     best_weight = (w1, w2, w3)

#     return best_weight, best_score

# def weighted_ensemble_three_prediction(
#         val_preds_cmpn,
#         val_preds_lgt,
#         val_preds_fm,
#         test_preds_cmpn,
#         test_preds_lgt,
#         test_preds_fm,
#         val_labels,
#         test_labels,
#         evaluator,
#         dataset_type,
#         label_mean, 
#         label_std
#     ):

#     avg_test_pred = (np.asarray(test_preds_cmpn) + np.asarray(test_preds_lgt) + np.asarray(test_preds_fm))/3.0

#     if dataset_type == 'regression':
#         avg_test_pred = (avg_test_pred - label_mean) / label_std

#     avg_test_score = evaluator.eval(np.asarray(test_labels), avg_test_pred)

#     print("=" * 60)
#     print("Avg. weight, Test score:", avg_test_score)
#     best_weight, best_weighted_val = search_best_weight_three(
#         val_preds_cmpn,
#         val_preds_lgt,
#         val_preds_fm,
#         val_labels,
#         evaluator,
#         dataset_type,
#         label_mean,
#         label_std,
#         step=0.05
#     )

#     w1, w2, w3 = best_weight

#     print("=" * 60)
#     print("Best weight:", best_weighted_val)
#     print("CMPN :", w1, "LiGhT:", w2, "Fusion:", w3)

#     weighted_test_pred = (w1 * np.asarray(test_preds_cmpn) + w2 * np.asarray(test_preds_lgt) + w3 * np.asarray(test_preds_fm))

#     if dataset_type == 'regression':
#         weighted_test_pred = (weighted_test_pred - label_mean) / label_std

#     weighted_test_score = evaluator.eval(np.asarray(test_labels), weighted_test_pred)

#     print("Best val :", best_val, "Test score:", weighted_test_score)

#     return avg_test_score, weighted_test_score, avg_test_pred, weighted_test_pred, best_weight



if __name__ == '__main__':

    # args = parse_args()
    # finetune(args)


    set_random_seed(2025)
    args = parse_args()
    print("args:", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # begin
    # --------------------------kelgt--------------------------
    config = config_dict[args.lgt_config]
    val_preds_lgt, test_preds_lgt, best_val, best_test, evaluator, result_tracker, train_loader, val_loader, test_loader, model_lgt, mean, std, train_index1, val_index1, test_index1 = run_lgt(args)
    print(f"val: {best_val:.6f}, test: {best_test:.6f}")
    result_single={'sacffold_seed':[args.seed], 'lgt_drp': [config['predict_drop']], 'lgt_lr': [args.lgt_lr], 'lgt_weight_decay': [args.lgt_weight_decay], 'val': [best_val], 'test': [best_test]}
    final_df = pd.DataFrame(result_single)

    print("args.result_save_path + '/result_KeLGT.csv':", args.result_save_path + '/result_KeLGT.csv')
    if os.path.isfile(args.result_save_path + '/result_KeLGT.csv'):
        final_df.to_csv(args.result_save_path + '/result_KeLGT.csv', mode='a', header=False, index=False)
    else:
        final_df.to_csv(args.result_save_path + '/result_KeLGT.csv', mode='a', index=False)
    # --------------------------end kelgt--------------------------

    # --------------------------cmpn--------------------------
    set_seed(args.seed)
    modify_train_args(args)

    args.init_lr = args.cmpnn_init_lr
    args.max_lr = args.cmpnn_max_lr
    args.final_lr = args.cmpnn_final_lr                 
    args.epochs = args.cmpnn_epochs
    args.warmup_epochs = args.cmpnn_warmup_epochs
    args.batch_size = args.cmpnn_batch_size
    args.data_path = args.data_path + '/' + args.dataset + '/' + args.dataset + '.csv'
    

    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    test_score_g, val_preds_cmpn, test_preds_cmpn, model_cmpn, train_data, val_data, test_data, train_index2, val_index2, test_index2 = run_cmpn(args, logger)
    final_result_single={'seed':[args.seed],'final_score':[test_score_g]}
    final_df = pd.DataFrame(final_result_single)
    if os.path.isfile(args.result_save_path + '/result_CMPN.csv'):
        final_df.to_csv(args.result_save_path + '/result_CMPN.csv', mode='a', header=False, index=False)
    else:
        final_df.to_csv(args.result_save_path + '/result_CMPN.csv', mode='a', index=False)


    assert train_index1==train_index2
    assert val_index1==val_index2
    assert test_index1==test_index2


    # stage3
    # ======================================== Representation-level Fusion ========================================
    print("========================================stage 3========================================") # stage3
    fusion_module = FeatureGatedResidualFusion(
            dim_g=300,          # CMPN embedding dim
            dim_l=768 * 3,      # line graph Transformer embedding dim
            hidden_dim=256,
            out_dim=args.num_tasks,
            dropout=0.0,# 0.1
        ).to(device)
    
    val_preds_fm, val_preds_cmpn, val_preds_lgt, test_preds_fm, val_preds_cmpn, val_preds_lgt,\
        best_val_fm, test_score_fm, val_labels, test_labels, mmm = train_fusion_stage(
            args=args,
            model_cmpn=model_cmpn,
            model_lgt=model_lgt,
            fusion_module=fusion_module,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            evaluator=evaluator,
            device=device,
            label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, 
            label_std=torch.from_numpy(std).to(device) if std is not None else None
        )

    print("best_val_fm, test_score_fm:", best_val_fm, test_score_fm)


    final_result_single={'seed':[args.seed], 'lgt_lr':[args.lgt_lr], 'test_predictions_LG':[best_test], 'test_predictions_G':[test_score_g], \
                    'test_score_fm':[test_score_fm]}
    final_df = pd.DataFrame(final_result_single)
    if os.path.isfile(args.result_save_path + '/final_fm_fusion_FGFR_result.csv') and args.seed!=1:
        final_df.to_csv(args.result_save_path + '/final_fm_fusion_FGFR_result.csv', mode='a', header=False, index=False)
    else:
        final_df.to_csv(args.result_save_path + '/final_fm_fusion_FGFR_result.csv', mode='a', index=False)





    