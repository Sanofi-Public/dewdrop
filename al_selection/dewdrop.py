"""
Retrospective Experiment #2: DEWDROP 
---
1. Produce ensembles of prediction for each batch and run PCA on each output ensemble 
2. Calculate the joint entropy of the ensembles 
3. Select the next batch with the calculated joint entropy and finetune on that batch
"""

import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os
import tqdm
import pickle
import time
from pathlib import Path
import shutil
from ast import literal_eval
import numpy as np

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, Subset

# To load alien from the neighbouring repo:
# import sys
# from pathlib import Path
# p = (Path(__file__).parent.parent / "UDS-active_learning_sdk").as_posix()
# print(p)
# sys.path.append(p)

from model_utils.models import NN
from utils.utils_data import collate_fn
from utils.align import align_ensemble

from alien.selection import EntropySelector, RandomSelector # input joint_entropy manually into _select
from alien.data import ObjectDataset # store the unlabeled dataset
from alien.stats import (joint_entropy_from_ensemble, 
                         joint_entropy_from_covariance, 
                         covariance_from_ensemble,  # use this to calculate joint entropy 
                         apply_pca) 
from alien.models import PytorchRegressor # predict ensemble 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def joint_entropy_from_ensemble_without_pca(
    preds,
    epsilon=None,
    rel_epsilon=1e-4,
    generate=True,
    ddof=1,
    block_size=None,
    pbar=False,
):
    return joint_entropy_from_covariance(
        covariance_from_ensemble(preds, generate=generate, ddof=ddof, block_size=block_size),
        epsilon=epsilon,
        rel_epsilon=rel_epsilon,
        generate=True,
        pbar=pbar,
    )


def dewdrop(args):     
    # Parse command line arguments
    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    training_config_path = CONFIG["training"]["training_config_path"]
    initial_weights = CONFIG["training"]["initial_weights"]
    config_path = CONFIG["training"]["model_config_path"]
    use_ensemble_cache = False if CONFIG["training"]["use_ensemble_cache"] == 'False' else True
    final_label_pool_size = int(CONFIG["dewdrop"]["final_labelpool_size"])
    
    # training configuration
    with open(training_config_path, "r") as f:
        training_config = json.load(f)
    
    batch_size = training_config['train_batch_size']
    ensemble_size = training_config['ensemble_size']
    pca_components = training_config['pca_components'] # or pick a smaller number than ensemble size to maintain rect. matrix 
   
    # model configuration
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


    # Validation Dataset loading
    with open(CONFIG["training"]["validation_data_path"], "rb") as f:
        validation_dataset = pickle.load(f)
    validation_loader = DataLoader(validation_dataset, 
                                   collate_fn=collate_fn, 
                                   num_workers=training_config["val_num_workers"])

    print("Printing Unlabeled pool and validation dataset sizes...")
    print(f"Unlabeled pool size: {len(train_dataset)}, validation_dataset size: {len(validation_dataset)}")
    print(f"Expected labeled pool size: {final_label_pool_size}")
    
    
    
    
    # Checkpoints
    # TODO: load model weights for script restart 
    model_dir = CONFIG["training"]["output_dir"]
    resume_from_checkpoint = None
    if training_config["restart_from_ckpt"]:
        resume_from_checkpoint = training_config["ckpt_file_path"]
        print(f"Restarting from checkpoint...{resume_from_checkpoint}")
    elif initial_weights != "":
        # Using original model weight
        print(f"Loading initial model weight...{initial_weights}")
        try:
            model_nn.load_state_dict(torch.load(initial_weights)["state_dict"], weights_only=False)
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
    
    # Restart from the right batch and unlabeled pool 
    unlabeled_pool = ObjectDataset(data=train_dataset.indices)
    labeled_pool = None
    batch_indx = 0
    
    record_path = Path(model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"] + "/record")
    record_path.mkdir(exist_ok=True)
    
    # Find the latest batch and unlabled/labeled indices 
    if any(record_path.iterdir()): 
        batch_indx = max([int(fp.name.strip(".npz")) for fp in record_path.iterdir()])
        loaddict = np.load(record_path / f"{batch_indx}.npz", allow_pickle=True)
        unlabeled_indices, labeled_indices = loaddict['unlabeled'], loaddict['labeled']
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
    
    # Load or generate droupout seed
    dropout_seeds_fp = os.path.join("models", "dewdrop", f"seeds_ensemble={ensemble_size}.npy")
    if os.path.exists(dropout_seeds_fp): 
        dropout_seeds = np.load(dropout_seeds_fp)
    else: 
        dropout_seeds = np.random.rand(ensemble_size)
        np.save(file=dropout_seeds_fp, arr=dropout_seeds) 


    while batch_indx < 1:
        
        ensembles_fp = Path('/'.join(CONFIG['training']['initial_weights'].split('/')[:-1])) / f"bs={batch_size}-ensembles_size={ensemble_size}-selection_round={batch_indx}.pt"
        
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


        # For DEWDROP, wrap this mode with PyTorchModel so we can produce ensembles 
        model_nn_alien = PytorchRegressor(
            model=model_nn.to(device),
            batch_size=training_config["train_batch_size"],
            iterate_inputs=False,
            trainer='lightning',
            stack_outputs=False, 
            stack_samples='outer', 
        )

        # Produce ensemble and calculate joint entropy
        X = DataLoader(Subset(dataset=train_dataset.dataset, indices=unlabeled_pool.data,), 
                        batch_size=training_config["train_batch_size"],
                        num_workers=training_config["train_num_workers"],
                        collate_fn=collate_fn,)
        
        start_ensemble = time.time()
        # Check if there is an available ensemble here already 
        print(ensembles_fp)
        if not os.path.exists(ensembles_fp):
        # or not use_ensemble_cache:

            # Iterate dataloader here and produce ensemble here
            # samples_shape = [`ensemble`: [`batch`: [`protein_in_batch`: Data['X_pred'] -> tensor(resnum, 9, 3)]]]
            # target_shape = [`batch`: [`protein_in_batch`: tensor(ensemble, protein_properties)]]
            ensembles = [] # store every ensemble, len(ensembles) == num_batches

            for x in tqdm.tqdm(X, desc="Process over batches..."): 

                x = x.to(device)
                ensemble = model_nn_alien.predict_samples(
                    X=x,
                    n=ensemble_size,
                    seeds=dropout_seeds,
                    use_lightning=False,
                    compute_loss=False, 
                    return_struct=True,
                )

                # Obtain the mask for prediction (output coord is [0, 0, 0])
                masks = [torch.all(torch.where(e == 0, 0, 1), dim=-1, keepdim=True) for e in ensemble]

                # Alignment of protein sequences
                ensemble = [align_ensemble(e) for e in ensemble]

                # Mask atoms that has coordinate (0, 0, 0)
                ensemble = [e * mask for e, mask in zip(ensemble, masks)]

                # Job getting killed, apply pca for each protein ensemble
                ensemble = [apply_pca(e.view(ensemble_size, -1), pca_components, sample_axis=0, top_variance=False) for e in ensemble]
                ensembles.extend(ensemble) # List[Tensor] -> (numProtein, [ensemble_size, pca_components]) 

            ensembles = torch.stack(ensembles, dim=0)

            # Save the ensemble and clear cuda cache 
            torch.save(ensembles, ensembles_fp)
            torch.cuda.empty_cache() 
            print("Generated ensembles and saved at '", ensembles_fp, "'.")
        else: 
            # load the ensemble for processing
            ensembles = torch.load(ensembles_fp, map_location=device) 
            print("Loaded existing ensembles from '", ensembles_fp, "'.")

        end_ensemble = time.time()

        # Calculate Joint Entropy on ensemble 
        jen = joint_entropy_from_ensemble_without_pca(
            preds=ensembles,
            epsilon=2e-3,
            block_size=50,
            pbar=True
        )

        # Select new batch based on the joint entropy matrix
        dewdrop_selector =  EntropySelector(
            batch_size=training_config["train_batch_size"],
            precompute_entropy=True,
            random_seed=42
        )
        new_batch_indices = dewdrop_selector.select(
            joint_entropy=jen.to("cpu"),
        ) # need to move to cpu because of calculation with prior

  
        if labeled_pool is None: 
            labeled_pool = ObjectDataset(unlabeled_pool[new_batch_indices])
        else: 
            labeled_pool.extend(unlabeled_pool[new_batch_indices])

        # Remove selected batch from unlabeled pool 
        unlabeled_pool.data = np.delete(
            arr=unlabeled_pool.data,
            obj=new_batch_indices,
            axis=0,
        )

        end_selection = time.time()

        duration_ensemble, duration_selection = (end_ensemble-start_ensemble)//60, (end_selection-end_ensemble)//60

        print(f"Time took to generate ensemble: {duration_ensemble}")
        print(f"Time took to select new batch: {duration_selection}")

        batch_indx += 1
        
        # Save current state of the unlabeled and labeled indices 
        np.savez(record_path / f"{batch_indx}.npz", unlabeled=unlabeled_pool.data, labeled=labeled_pool.data)

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
            # num_workers=training_config["train_num_workers"],
            num_workers=0, 
            collate_fn=collate_fn,
            pin_memory=training_config["train_pin_memory"],
        )
        

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
    # print("Total test time: %.2f sec" % timer_callback.time_elapsed("test"))


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())
    dewdrop(args)