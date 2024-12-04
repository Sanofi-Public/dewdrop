"""
Retrospective Experiment #1: Random Selection
---
1. Select a random batch to perform finetuneing. 
2. Repeat the first step by concatenating a new batch with the existing batch until exhausting all the training dataset. 
3. Other than loss performance, also benchmark time performance at each step. (Make sure to use early stopping)
"""

import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os
import pickle
from pathlib import Path
import math
import shutil
from ast import literal_eval
import numpy as np 

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset

from model_utils.models import NN
from utils.utils_data import collate_fn

from alien.selection import RandomSelector
from alien.data import ObjectDataset, ArrayDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())

    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])
    
    training_config_path = CONFIG["training"]["training_config_path"]
    initial_weights = CONFIG["training"]["initial_weights"]
    config_path = CONFIG["training"]["model_config_path"]
    final_label_pool_size = int(CONFIG["dewdrop"]["final_labelpool_size"])

    with open(training_config_path, "r") as f:
        training_config = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    model_nn = NN(**config)

    # unfreeze layers(rest freezed)
    # Note: Layer num start from 0(i.e. first layer will be layer 0)
    print(
        "Unfrozen layers: ",
        training_config["unfreeze_layer_num"],
        "; An empty list means all layers are trainable."
    )
    if len(training_config["unfreeze_layer_num"]) > 0:
        model_nn.freeze_layer(training_config["unfreeze_layer_num"])

    model_nn = model_nn.to(device)

    # Train Dataset loading
    with open(CONFIG["training"]["train_data_path"], "rb") as f:
        train_dataset = pickle.load(f)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["train_batch_size"],
        drop_last=training_config["train_drop_last"],
        shuffle=training_config["train_shuffle"],
        num_workers=training_config["train_num_workers"],
        collate_fn=collate_fn,
        pin_memory=training_config["train_pin_memory"],
    )

    # Validation Dataset loading
    with open(CONFIG["training"]["validation_data_path"], "rb") as f:
        validation_dataset = pickle.load(f)
    validation_loader = DataLoader(validation_dataset, 
                                   collate_fn=collate_fn, 
                                   num_workers=training_config["val_num_workers"])

    print("Printing train and validation dataset sizes...")
    print(f"train_dataset size: {len(train_dataset)}, validation_dataset size: {len(validation_dataset)}")
    
    # Checkpoints
    model_dir = CONFIG["training"]["output_dir"]
    resume_from_checkpoint = None
    if training_config["restart_from_ckpt"]:
        resume_from_checkpoint = training_config["ckpt_file_path"]
        print(f"Restarting from checkpoint...{resume_from_checkpoint}")
    elif initial_weights != "":
        # Using original model weight
        print(f"Loading initial model weight...{initial_weights}")
        try:
            model_nn.load_state_dict(torch.load(initial_weights, weights_only=False)["state_dict"])
        except:
            model_nn.load_state_dict(torch.load(initial_weights, weights_only=False))

    # Loggers
    Path(model_dir).mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"] + "/checkpoints",
        every_n_train_steps=training_config["checkpoint_steps"],
        save_top_k=training_config["checkpoint_limit"],
        save_last=training_config["save_last"],
        save_on_train_epoch_end=training_config["save_on_train_epoch_end"]
    )
    
    timer_callback = pl.callbacks.Timer()

    # Dump all the config files in the save directory
    Path(model_dir + "/" + CONFIG["training"]["run_name"]).mkdir(exist_ok=True)
    Path(model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"]).mkdir(exist_ok=True)
    shutil.copy2(args["config"], model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"])
    shutil.copy2(CONFIG["training"]["training_config_path"], model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"])
    shutil.copy2(CONFIG["training"]["model_config_path"], model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"])

    # ADD: tackle the case where no initial weights are provided
    if initial_weights != "":
        shutil.copy2(initial_weights, model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"])
    
    # Build the labeled dataset (for finetuning) by iterating over the dataloader 
    # train_num_batches = int(math.ceil(len(train_dataset)/training_config['train_batch_size']))
    unlabeled_pool = ObjectDataset(data=train_dataset.indices)
    labeled_pool = None
    batch_indx = 0

    record_path = Path(model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"] + "/record")
    record_path.mkdir(exist_ok=True)

    # Find the latest batch and unlabled/labeled indices 
    if any(record_path.iterdir()): 
        batch_indx = max([int(fp.name.strip(".npz")) for fp in record_path.iterdir()])
        loaddict = np.load(record_path / f"{batch_indx}.npz")
        unlabeled_indices, labeled_pool = loaddict['unlabeled'], loaddict['labeled']
        unlabeled_pool = ObjectDataset(data=unlabeled_indices)
        labeled_pool = ObjectDataset(data=labeled_indices)
        # reload the partially trained weight 
        partial_weight = model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"]
        try:
            model_nn.load_state_dict(torch.load(partial_weight)["state_dict"])
        except:
            model_nn.load_state_dict(torch.load(partial_weight))

    # Build the labeled dataset (for finetuning) by iterating over the dataloader 
    num_batches = int(final_label_pool_size/training_config['train_batch_size'])
    print("Total number of batch selection rounds: ", num_batches)
    print(f"Current selection round: {batch_indx}/{num_batches}")

    selector = RandomSelector(
        model=None,
        samples=unlabeled_pool,
        batch_size=training_config["train_batch_size"],
        random_seed=42
    )

    while batch_indx < num_batches:
        
        # Tensorboard logger
        tb_logger = TensorBoardLogger(
            save_dir=model_dir,
            name=CONFIG["training"]["run_name"]+f"_batches={batch_indx}",
            version=CONFIG["training"]["version_id"],
        )
        
        csv_logger = CSVLogger(
            save_dir=model_dir,
            name=CONFIG["training"]["run_name"]+f"_batches={batch_indx}",
            version=CONFIG["training"]["version_id"],
        )


        # Select new batches and remove from unlabeled pool 
        new_batch_indices = selector.select(return_indices=True)
        if labeled_pool is None: 
            labeled_pool = ObjectDataset(data=unlabeled_pool[new_batch_indices])
        else: 
            labeled_pool.extend(unlabeled_pool[new_batch_indices])
        unlabeled_pool.data = np.delete(
            arr=unlabeled_pool.data,
            obj=new_batch_indices,
            axis=0,
        )

        # Curate new training dataloader
        train_subset = Subset(
            dataset=train_dataset.dataset, 
            indices=labeled_pool.data,
        )
        train_subloader = DataLoader(
            train_subset,
            batch_size=training_config["train_batch_size"],
            drop_last=training_config["train_drop_last"],
            shuffle=training_config["train_shuffle"],
            num_workers=training_config["train_num_workers"],
            collate_fn=collate_fn,
            pin_memory=training_config["train_pin_memory"],
        )

        batch_indx += 1    

        """Model training and fitting"""
        # ADD: early stopping mechanism
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20)

        trainer = pl.Trainer(
            max_epochs=training_config["max_epochs"],
            strategy=training_config["strategy"],
            logger=(tb_logger, csv_logger),
            log_every_n_steps=training_config["log_every_n_steps"],
            default_root_dir=model_dir,
            callbacks=[checkpoint_callback, timer_callback, early_stopping],
            devices=literal_eval(CONFIG.get("training", "devices")) if CONFIG.get("training", "devices")!='' else 'auto'
        )
        print("Trainer assigned! Start fitting!")

        trainer.fit(
            model=model_nn,
            train_dataloaders=train_subloader,
            val_dataloaders=validation_loader,
            ckpt_path=resume_from_checkpoint,
        )
        print("Done fitting!")

        print("Start Validation")
        trainer.validate(model=model_nn, dataloaders=validation_loader)

    # if CONFIG.get("training",'test_data_path')!='':
    #     # Test Dataset loading
    #     with open(CONFIG["training"]["test_data_path"], "rb") as f:
    #         test_dataset = pickle.load(f)
    #     print("Start Testing")
    #     test_loader = DataLoader(test_dataset, collate_fn=collate_fn)
    #     trainer.test(model=model_nn, dataloaders=test_loader)
    
    print("Total training time: %.2f sec" % timer_callback.time_elapsed("train"))
    print("Total validation time: %.2f sec" % timer_callback.time_elapsed("validate"))
    print("Total test time: %.2f sec" % timer_callback.time_elapsed("test"))
