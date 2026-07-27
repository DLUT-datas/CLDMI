import warnings
warnings.filterwarnings('ignore')
# from rdkit import RDLogger  
# RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple

import numpy as np

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.utils import create_logger
from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
import random
import torch
import pandas as pd
import time


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



def hold_out(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """hold-out"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    # makedirs(args.save_dir)
    task_names = get_task_names(args.data_path)
    print("task_names:", task_names)

    # Run training on different random seeds for each hold-out
    all_scores = []
    for run_num in range(args.num_runs): # 一个种子运行一次
 
        # args.save_dir = os.path.join(save_dir, f'hold_out_{run_num}')
        args.save_dir = os.path.join(save_dir, f'hold_out_{init_seed}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
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

    return mean_score, std_score





if __name__ == '__main__':
    
    score_lst = []
    
    args = parse_train_args()

    args.gpu = 0
    args.epochs = 30
    args.ensemble_size = 1
    args.batch_size = 64
    args.split_sizes = [0.8, 0.1, 0.1] #
    args.num_runs = 1

    set_seed(args.seed)

    modify_train_args(args)
    result_path = args.save_dir + '/final_result.csv'
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    begin_time = time.time()
    mean_auc_score, std_auc_score = hold_out(args, logger)
    end_time = time.time()
    run_time = round(end_time-begin_time)
    # 计算时分秒
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    # 输出
    print (f'该程序运行时间：{hour}小时{minute}分钟{second}秒')
    score_lst.append(mean_auc_score)

    final_result_single={'seed':[args.seed],'final_score':[mean_auc_score]}
    final_df = pd.DataFrame(final_result_single)
    final_df.to_csv(result_path, mode='a',header=False, index=False)
