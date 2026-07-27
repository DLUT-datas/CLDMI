from typing import List

import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler
import numpy as np


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            # batch_preds = model(batch, features_batch)
            batch_preds, batch_embs = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds


def predict_with_embeddings(
    model: nn.Module,
    data: MoleculeDataset,
    batch_size: int,
    scaler: StandardScaler = None
):
    model.eval()

    preds = []
    embs = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch = mol_batch.smiles()
        features_batch = mol_batch.features()

        with torch.no_grad():
            batch_preds, batch_embs = model(smiles_batch, features_batch)

        batch_preds = batch_preds.detach().cpu().numpy()
        batch_embs = batch_embs.detach().cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        preds.extend(batch_preds.tolist())
        embs.append(batch_embs)

    embs = np.concatenate(embs, axis=0)

    return preds, embs


def evaluate_dataloader2(model: nn.Module, dataloader, evaluator, scaler: StandardScaler = None) -> List[List[float]]:
    model.eval()

    predictions_all, embs_all, labels_all = [], [], []
    for batched_data in dataloader:
        smiles_batch, g_batch, ecfp_batch, md_batch, labels_batch = batched_data

        with torch.no_grad():
            # batch_preds = model(batch, features_batch)
            batch_preds, batch_embs = model(smiles_batch, None)

            batch_preds = batch_preds.detach().cpu()
            batch_embs = batch_embs.detach().cpu()
            labels_batch = labels_batch.detach().cpu()

        print("batch_preds:", type(batch_preds), "batch_embs:", type(batch_embs))
        # batch_preds = batch_preds.data.cpu().numpy()
        # batch_embs = batch_embs.data.cpu().numpy()
        # print("batch_preds:", type(batch_preds))

        # Inverse scale if regression
        if scaler is not None:
            batch_preds_np = batch_preds.numpy()
            batch_preds = scaler.inverse_transform(batch_preds_np)
            batch_preds = torch.tensor(batch_preds_np, dtype=torch.float64)

        # Collect vectors
        # batch_preds = batch_preds.tolist()
        predictions_all.append(batch_preds)
        embs_all.append(batch_embs)
        labels_all.append(labels_batch.detach().cpu())

    result = evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))

    return result, predictions_all, embs_all, labels_all

