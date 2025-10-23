import os
from typing import List, Dict, Tuple, Union, Optional, Any
import shutil
import inspect
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import copy
import yaml
import re
import dill
import numpy as np

from arms.cores.MassEntity.msentity.core.MSDataset import MSDataset
from arms.cores.MassMolKit.mmkit.fragment.CleavagePatternLibrary import CleavagePatternLibrary
from arms.cores.MassMolKit.mmkit.chem.Compound import Compound

from .utils.training_setup import *
from .utils.CheckPointManager import CheckPointManager
from .utils.EarlyStopping import EarlyStopping
from arms.specgen.ArmsSpecGen import ArmsSpecGen

DATASET_NAME = 'fragment_pathway'

loss_keys = ['fragment_loss', 'mol_loss']

def main(
        model_config: Dict[str, Any],
        experiment_dir: str,
        ckpt_id: str,
        device: torch.device,
        epoch: int,
        save_interval: int,
        ms_dataset: MSDataset,
        extra_data: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_info: Dict[str, Any],
        early_stopping_info: Dict[str, Any],
        run_dir: str,
):
    ckpt_manager = CheckPointManager(
        ckpt_root_dir=experiment_dir,
    )
    ckpt_manager.set_run_dir(run_dir)


    if ckpt_id is None or ((ckpt_id in ckpt_manager.ckpt_nodes) and ckpt_manager.ckpt_nodes[ckpt_id].is_base):
        if ckpt_id is None:
            ckpt_id = -1
        ckpt_manager.checkout_new_branch(root_ckpt_node_id=ckpt_id)
        initial_epoch = 1
        max_epoch = epoch
        global_step = 0

        model_config['params']['cleavage_scorer_params']['cleavage_pattern_lib_dict'] = extra_data['cleavage_pattern_lib_dict']
        model = ArmsSpecGen.from_params(model_config['params'])

        optimizer, scheduler = get_optimizer(model, optimizer_info, is_return_scheduler=True)
    else:
        ckpt_node = ckpt_manager.load_ckpt(ckpt_id)
        metrics_df = ckpt_manager.read_metrics()
        for _, row in metrics_df.iterrows():
            _epoch = int(row['epoch'])

            # Extract loss values
            train_loss = float(row['train_loss'])
            train_loss_dict = {k: float(row[f'train_{k}']) for k in loss_keys}
            val_loss = float(row['val_loss'])
            val_loss_dict = {k: float(row[f'val_{k}']) for k in loss_keys}
            lr = float(row['lr'])
            ckpt_manager.summary_writer.add_scalars("Loss", {
                "train": train_loss,
                "val": val_loss
            }, _epoch)
            for k in loss_keys:
                ckpt_manager.summary_writer.add_scalars(
                    f'{k}', {
                        "train": train_loss_dict[k],
                        "val": val_loss_dict[k]
                    }, _epoch
                )

            # Learning Rate
            lr = float(row['lr'])
            ckpt_manager.summary_writer.add_scalar("LR", lr, _epoch)
        ckpt_manager.summary_writer.flush()

        model, initial_epoch, global_step, optimizer, scheduler, optimizer_info, extra_data = ckpt_node.load_model(ArmsSpecGen.from_params, device=device)

        max_epoch = initial_epoch + epoch
        initial_epoch += 1
        
    unique_smiles_list = extra_data['unique_smiles']
    unique_adduct_list = extra_data['unique_adducts']
    model.to(device)

    ckpt_manager.initialize_metrics(columns=["epoch", "iter", "train_loss"]+[f'train_{k}' for k in loss_keys]+['val_loss']+[f'val_{k}' for k in loss_keys]+['lr'])
    # ckpt_manager.initialize_metrics(columns=["epoch", "iter", "train_loss", "train_ce_loss", "train_uncertain_loss", "val_loss", "val_ce_loss", "val_uncertain_loss", "train_acc", "val_acc", "lr"])

    early_stopping = EarlyStopping.from_params(early_stopping_info)

    for epoch in range(initial_epoch, max_epoch+1):
        iterator = tqdm(train_loader, desc=f"Train({epoch}/{max_epoch})")

        total_loss = 0.0
        total_loss_dict = defaultdict(float)
        total_samples = 0
        for batch in iterator:
            index = batch['index'].to(device)
            smiles_idx = batch['smiles_idx'].to(device)
            adduct_idx = batch['adduct_idx'].to(device)
            samples = index.size(0)

            smiles_list = [unique_smiles_list[i.item()] for i in smiles_idx]
            adduct_list = [unique_adduct_list[i.item()] for i in adduct_idx]

            for i in range(samples):
                compound = Compound.from_smiles(smiles_list[i])
                mol_graph, cleavage_scores = model(compound)

            total_samples += samples

            global_step += 1


        avg_train_loss = total_loss / total_samples
        avg_train_loss_dict = {k: v / total_samples for k, v in total_loss_dict.items()}

def prepare_train(project_dir: str, train_config_path: str, root_run_dir: str = None):
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    load_name = train_config['load_name']
    if load_name == '' or load_name is None:
        load_name = 'experiments' + datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir = os.path.join(project_dir, load_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    ckpt_id = train_config['ckpt_id']
    batch_size = train_config['batch_size']
    device = torch.device(train_config['device'])
    epoch = train_config['epoch']
    save_interval = train_config['save_interval']

    optimizer_info = train_config['optimizer']

    early_stopping_info = train_config['early_stopping']

    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    if root_run_dir is None:
        run_dir = os.path.join(experiment_dir, 'runs', now_str)
    else:
        run_dir = os.path.join(root_run_dir, now_str)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(train_config_path, os.path.join(run_dir, 'train_config.yaml'))

    return experiment_dir, ckpt_id, batch_size, device, epoch, save_interval, optimizer_info, early_stopping_info, run_dir


def extract_unique_indices(ms_dataset:MSDataset, smiles_col_name: str, adduct_col_name: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Extract unique SMILES and AdductType values and their corresponding index mappings.

    Args:
        ms_dataset (MSDataset): The mass spectrometry dataset.

    Returns:
        tuple:
            - unique_smiles (List[str]): List of unique SMILES strings.
            - smiles_indices (torch.Tensor): Tensor of shape (num_data,), each value = index of SMILES.
            - unique_adducts (List[str]): List of unique AdductType strings.
            - adduct_indices (torch.Tensor): Tensor of shape (num_data,), each value = index of AdductType.
    """
    meta = ms_dataset.meta

    # --- SMILES ---
    unique_smiles = meta[smiles_col_name].dropna().unique().tolist()
    smiles_index_map = {s: i for i, s in enumerate(unique_smiles)}
    smiles_indices = torch.tensor(
        meta[smiles_col_name].map(smiles_index_map).fillna(-1).astype(int).to_numpy(),
        dtype=torch.long
    )

    # --- AdductType ---
    unique_adducts = meta[adduct_col_name].dropna().unique().tolist()
    adduct_index_map = {a: i for i, a in enumerate(unique_adducts)}
    adduct_indices = torch.tensor(
        meta[adduct_col_name].map(adduct_index_map).fillna(-1).astype(int).to_numpy(),
        dtype=torch.long
    )

    return unique_smiles, smiles_indices, unique_adducts, adduct_indices

def setup_dataset(dataset_config_path: str, experiment_dir: str, batch_size: int, device: torch.device):
    if not os.path.exists(os.path.join(experiment_dir, 'ds', DATASET_NAME)) and not (dataset_config_path == '' or dataset_config_path is None):
        # Load configurations
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        print(f"No dataset found in {experiment_dir}, building a new dataset.")
        ms_dataset = MSDataset.from_hdf5(dataset_config['train_dataset_path'])
        smiles_col_name = dataset_config.get('smiles_col_name', 'SMILES')
        adduct_col_name = dataset_config.get('adduct_col_name', 'AdductType')
        unique_smiles, smiles_indices, unique_adducts, adduct_indices = extract_unique_indices(ms_dataset, smiles_col_name, adduct_col_name)
        cleavage_pattern_library = CleavagePatternLibrary.load_json(dataset_config['cleavage_pattern_lib_path'])

        variables = {
            'smiles_idx': smiles_indices,
            'adduct_idx': adduct_indices
        }
        extra_data = {
            'unique_smiles': unique_smiles,
            'unique_adducts': unique_adducts,
            'cleavage_pattern_lib_dict': cleavage_pattern_library.to_dict(),
        }

        dataset, train_dataloader, val_dataloader, test_dataloader = get_ds(
            variables, # [m15_id]
            train_size=dataset_config['train_size'],
            val_size=dataset_config['val_size'],
            test_size=dataset_config['test_size'],
            batch_size=batch_size,
            mode='train', device=device)
        
        dataset_dir = save_dataset(
            experiment_dir, dataset, train_dataloader, val_dataloader, test_dataloader,
            name=DATASET_NAME,
            extra_data=extra_data
        )
        ms_dataset_file_path = os.path.join(dataset_dir, 'ms_dataset.hdf5')
        ms_dataset.to_hdf5(ms_dataset_file_path)

        with open(os.path.join(dataset_dir, "dataset_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_config, f, sort_keys=False, allow_unicode=True)
    elif not os.path.exists(os.path.join(experiment_dir, 'ds', DATASET_NAME)):
        raise FileNotFoundError(f"Dataset not found in {experiment_dir} and no dataset_config_path provided.")

    dataset, train_dataloader, val_dataloader, test_dataloader, extra_data = load_dataset(
        load_dir=experiment_dir,
        name=DATASET_NAME,
        batch_size=batch_size,
        load_extra_data=True
    )
    # unique_smiles = extra_data['unique_smiles']
    # unique_adducts = extra_data['unique_adducts']
    ms_dataset = MSDataset.from_hdf5(os.path.join(experiment_dir, 'ds', DATASET_NAME, 'ms_dataset.hdf5'))

    return dataset, train_dataloader, val_dataloader, test_dataloader, ms_dataset, extra_data


# python -m training.train_arms -project data/training -model training/arms_specgen_configs/model.yaml -ds training/arms_specgen_configs/dataset_config.yaml -train training/arms_specgen_configs/train_config.yaml
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parameter tuning')
    parser.add_argument("-project", "--project_dir", type = str, default='', help = "Project directory")
    parser.add_argument("-model", "--model_config_path", type = str, default='', help = "Model configuration file (.yaml)")
    parser.add_argument("-ds", "--dataset_config_path", type = str, default='', help = "Dataset configuration file (.yaml)")
    parser.add_argument("-train", "--train_config_path", type = str, default='', help = "Training configuration file (.yaml)")


    args = parser.parse_args()
    project_dir = args.project_dir
    model_config_path = args.model_config_path
    dataset_config_path = args.dataset_config_path
    train_config_path = args.train_config_path

    experiment_dir, ckpt_id, batch_size, device, epoch, save_interval, optimizer_info, early_stopping_info, run_dir = prepare_train(project_dir, train_config_path)

    dataset, train_dataloader, val_dataloader, test_dataloader, ms_dataset, extra_data = setup_dataset(dataset_config_path, experiment_dir, batch_size, device)

    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    main(
        model_config=model_config,
        experiment_dir=experiment_dir,
        ckpt_id=ckpt_id,
        device=device,
        epoch=epoch,
        save_interval=save_interval,
        ms_dataset=ms_dataset,
        extra_data=extra_data,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer_info=optimizer_info,
        early_stopping_info=early_stopping_info,
        run_dir=run_dir,
    )