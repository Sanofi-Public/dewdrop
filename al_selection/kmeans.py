"""
Retrospective Experiment #4: Kmeans  
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

from model_utils.models import NN as NN_grad
from utils.utils_data import collate_fn

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def extract_embedding(full_dataset, indices, model): 
    """
    Pass data through the model and generate a embedding with constant size to be clustered.
    """
    embedding_ensemble = []
    dataset = Subset(full_dataset, torch.as_tensor(indices))
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        drop_last=False, 
        shuffle=False,
        num_workers=0, 
        collate_fn=collate_fn, 
        pin_memory=True, 
    )
    model.eval()
    for data in tqdm.tqdm(dataloader, desc="Generate ensembles"): 
        data = data.to(device)
        model.zero_grad()
        return_dict = model(data)
        s = return_dict['final_block_emb']
        s = s.to('cpu')
        embedding_ensemble.append(s.view(-1))
    # pad the embeddings to fix length
    maxlen = max(len(s) for s in embedding_ensemble)
    for si in range(len(embedding_ensemble)): 
        p1d = (0, maxlen-len(embedding_ensemble[si]))
        if p1d[1] != 0: 
            embedding_ensemble[si] = torch.nn.functional.pad(embedding_ensemble[si], p1d, 'constant', 0.0)
    return torch.stack(embedding_ensemble).cpu().detach().numpy()

 

def kmeans_sample_batch(unlabeled_embedding, indices, K, random_state=10): 
    """
    Return a selected batch with size K given the gradient ensemble and the dataset.
    """
#     assert len(unlabeled_pool) == len(indices), "gradient ensemble size doesn't match the input dataset size."
    print("Clustering the ensembles...")
    kmeans = KMeans(n_clusters=K, random_state=10, init='k-means++', n_init="auto").fit(unlabeled_embedding)
    centroids = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centroids, unlabeled_embedding)
    selected_data_instances = [indices[i] for i in closest]
    print("Finished clustering!")
    assert len(selected_data_instances) == K, "output batch doesn't have size equal to K."
    return np.stack(selected_data_instances), closest

def kmeans(args):     
    # Parse command line arguments
    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    training_config_path = CONFIG["training"]["training_config_path"]
    initial_weights = CONFIG["training"]["initial_weights"]
    config_path = CONFIG["training"]["model_config_path"]

    with open(training_config_path, "r") as f:
        training_config = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    # model_nn = NN(**config)
    model_grad = NN_grad(**config)

    # unfreeze layers(rest freezed)
    # Note: Layer num start from 0(i.e. first layer will be layer 0)
    print(
        "Unfrozen layers: ",
        training_config["unfreeze_layer_num"],
        "; An empty list means all layers are trainable."
    )
    if len(training_config["unfreeze_layer_num"]) > 0:
        # model_nn.freeze_layer(training_config["unfreeze_layer_num"])
        model_grad.freeze_layer(training_config["unfreeze_layer_num"])

    # model_nn = model_nn.to(device)
    model_grad = model_grad.to(device)

    # Train Dataset loading
    with open(CONFIG["training"]["train_data_path"], "rb") as f:
        train_dataset = pickle.load(f)

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
            # model_nn.load_state_dict(torch.load(initial_weights)["state_dict"])
            model_grad.load_state_dict(torch.load(initial_weights, weights_only=False)["state_dict"])
        except:
            # model_nn.load_state_dict(torch.load(initial_weights))
            model_grad.load_state_dict(torch.load(initial_weights, weights_only=False))

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
    num_batches = int(len(train_dataset)/training_config['train_batch_size'])

    print("Total number of batch selection rounds: ", num_batches)

    unlabeled_pool = train_dataset.indices
    labeled_pool = None
    batch_indx = 0
    
    # Hyperparameter
    batch_size = training_config['train_batch_size']
    ensemble_size = training_config['ensemble_size']

    while batch_indx < num_batches:
        ensembles_fp = os.path.join("models", "badge", f"bs={batch_size}-selection_round={batch_indx}.pt")

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

        # Kmeans sample new batch (use relative_indices to remove labeled data from unlabel pool)
        start_selection = time.time()
        
        unlabeled_embedding = extract_embedding(
            train_dataset.dataset, 
            unlabeled_pool,
            model_grad.to(device)
        )
        
        new_batch_indices, relative_indices = kmeans_sample_batch(
            unlabeled_embedding, 
            unlabeled_pool, 
            training_config["train_batch_size"]
        )

        if labeled_pool is None: 
            labeled_pool = new_batch_indices
        else: 
            labeled_pool = np.concatenate([labeled_pool, new_batch_indices])

        # Remove selected batch from unlabeled pool 
        unlabeled_pool = np.delete(
            arr=unlabeled_pool,
            obj=relative_indices,
            axis=0,
        )

        end_selection = time.time()
        duration_selection = (end_selection-start_selection)//60 
        print(f"Time took to select new batch: {duration_selection}")


        batch_indx += 1

        train_subset = Subset(train_dataset.dataset, torch.as_tensor(labeled_pool))
        train_subloader = DataLoader(
            train_subset,
            batch_size=training_config["train_batch_size"],
            drop_last=training_config["train_drop_last"],
            shuffle=training_config["train_shuffle"],
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=training_config["train_pin_memory"],
        ) # subset must set num_worker=0 to make the training on following batches work. Weird bug with PL
        

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
            model=model_grad,
            train_dataloaders=train_subloader,
            val_dataloaders=validation_loader,
            ckpt_path=resume_from_checkpoint,
        )
        print("Done fitting!")

        print("Start Validation")
        trainer.validate(model=model_grad, dataloaders=validation_loader)

    if CONFIG.get("training",'test_data_path')!='':
        # Test Dataset loading
        # with open(CONFIG["training"]["test_data_path"], "rb") as f:
        #     test_dataset = pickle.load(f)
        print("Start Testing")
        test_loader = DataLoader(test_dataset, collate_fn=collate_fn)
        trainer.test(model=model_grad, dataloaders=test_loader)
    
    print("Total training time: %.2f sec" % timer_callback.time_elapsed("train"))
    print("Total validation time: %.2f sec" % timer_callback.time_elapsed("validate"))
    print("Total test time: %.2f sec" % timer_callback.time_elapsed("test"))


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())
    kmeans(args)