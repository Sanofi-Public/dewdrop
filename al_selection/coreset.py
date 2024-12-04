"""
Retrospective Experiment #5: CORESET 
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

from sklearn.metrics import pairwise_distances

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, Subset

# To load alien from the neighbouring repo:
import sys
from pathlib import Path
p = (Path(__file__).parent.parent / "UDS-active_learning_sdk").as_posix()
print(p)
sys.path.append(p)

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

class Coreset_Greedy:
    """
    Given the complete ensemble pool, sample new batch iteratively in a greedy manner. 
    """
    def __init__(self, all_ensembles):
        self.all_ensembles = np.array(all_ensembles)
        self.dset_size = len(all_ensembles)
        self.min_distances = None
        self.already_selected = []

        # reshape
        # feature_len = self.all_ensembles[0].shape[1]
        # self.all_ensembles = self.all_ensembles.reshape(-1, feature_len)


    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]
        
        if centers is not None:
            x = self.all_ensembles[centers] # pick only centers
            # print(x.shape)
            # print(self.all_ensembles.shape)
            dist = pairwise_distances(self.all_ensembles, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
    
    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        new_batch = []
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)
            
            # assert ind not in already_selected
            self.update_dist([ind],only_new=True, reset_dist=False)
            new_batch.append(ind)
        
        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance.item())

        return new_batch, max_distance

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

def coreset(args):     
    # Parse command line arguments
    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    training_config_path = CONFIG["training"]["training_config_path"]
    initial_weights = CONFIG["training"]["initial_weights"]
    config_path = CONFIG["training"]["model_config_path"]
    final_label_pool_size = int(CONFIG["dewdrop"]["final_labelpool_size"])
    
    # training configuration
    with open(training_config_path, "r") as f:
        training_config = json.load(f)
    
    batch_size = training_config["train_batch_size"]
    ensemble_size = training_config['ensemble_size']
    pca_components = training_config['pca_components'] 
   
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

    # (TODO: Uncomment after public release)
    # # Train Dataset loading
    # with open(CONFIG["training"]["train_data_path"], "rb") as f:
    #     train_dataset = pickle.load(f)


    # # Validation Dataset loading
    # with open(CONFIG["training"]["validation_data_path"], "rb") as f:
    #     validation_dataset = pickle.load(f)
    # validation_loader = DataLoader(validation_dataset, 
    #                                collate_fn=collate_fn, 
    #                                num_workers=training_config["val_num_workers"])

    print("Printing Unlabeled pool and validation dataset sizes...")
    print(f"Unlabeled pool size: {len(train_dataset)}, validation_dataset size: {len(validation_dataset)}")
    print(f"Expected labeled pool size: {final_label_pool_size}")
    
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
    
    # use labeled_pool to keep track of selected ensembles 
    labeled_pool = []
    batch_indx = 0
    
    record_path = Path(model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"] + "/record")
    record_path.mkdir(exist_ok=True)
    
    # Find the latest batch and unlabled/labeled indices 
    if any(record_path.iterdir()): 
        batch_indx = max([int(fp.name.strip(".npz")) for fp in record_path.iterdir()])
        loaddict = np.load(record_path / f"{batch_indx}.npz")
        labeled_pool = loaddict['labeled'].tolist()
        # reload the partially trained weight 
        partial_weight = model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"] + '/weights_bs=' + str(batch_size) + "_" + CONFIG["training"]["version_id"] + '.pt' 
        try:
            model_nn.load_state_dict(torch.load(partial_weight)["state_dict"])
        except:
            model_nn.load_state_dict(torch.load(partial_weight))

    
    # Build the labeled dataset (for finetuning) by iterating over the dataloader 
    num_batches = int(final_label_pool_size/training_config['train_batch_size'])
    print("Total number of batch selection rounds: ", num_batches)

    # Generate ensemble for the entire dataset 
    start_ensemble = time.time()
    ensembles_fp = Path(model_dir + "/" + CONFIG["training"]["run_name"] + "/" + CONFIG["training"]["version_id"]) / f"bs={batch_size}-ensembles_size={ensemble_size}-selection_round={batch_indx}.pt"
    ensembles_full_dataset = None
    if not os.path.exists(ensembles_fp): 
        ensembles_full_dataset = extract_embedding(train_dataset.dataset, train_dataset.indices, model_nn)
    if ensembles_full_dataset is not None:
        # Save the ensemble and clear cuda cache 
        torch.save(ensembles_full_dataset, ensembles_fp)
        print("Generated ensembles and saved at '", ensembles_fp, "'.")
    else: 
        # load the ensemble for processing
        ensembles_full_dataset = torch.load(ensembles_fp, map_location=device) 
        print("Loaded existing ensembles from '", ensembles_fp, "'.")
    end_ensemble = time.time()

    # randomly select a point as starting point 
    labeled_pool.append(np.random.randint(0, len(ensembles_full_dataset)))

    # Create CORESET object 
    coreset_obj = Coreset_Greedy(all_ensembles=ensembles_full_dataset)

    while batch_indx < num_batches:
        print(f"Current selection round: {batch_indx+1}/{num_batches}")

        start_selection = time.time()

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
        
        # sample from coreset_obj 
        if batch_indx == 0:
            new_indices, max_distance = coreset_obj.sample(already_selected=labeled_pool, sample_size=batch_size-1)
        else: 
            new_indices, max_distance = coreset_obj.sample(already_selected=labeled_pool, sample_size=batch_size)
        labeled_pool.extend(new_indices)

        end_selection = time.time()

        duration_ensemble, duration_selection = (end_ensemble-start_ensemble)//60, (end_selection-start_selection)//60

        print(f"Time took to generate ensemble: {duration_ensemble}")
        print(f"Time took to select new batch: {duration_selection}")
        
        # Save current state of the unlabeled and labeled indices 
        np.savez(record_path / f"{batch_indx}.npz", labeled=labeled_pool)

        # Curate new training dataloader
        train_subset = Subset(
            dataset=train_dataset.dataset, 
            indices=labeled_pool,
        )
        train_subloader = DataLoader(
            train_subset,
            batch_size=training_config["train_batch_size"],
            drop_last=training_config["train_drop_last"],
            shuffle=training_config["train_shuffle"],
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=training_config["train_pin_memory"],
        )
        

        """Model training and fitting"""
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
            model=model_nn.to(device),
            train_dataloaders=train_subloader,
            val_dataloaders=validation_loader,
            ckpt_path=resume_from_checkpoint,
        )
        print("Done fitting!")

        print("Start Validation")
        trainer.validate(model=model_nn, dataloaders=validation_loader)

        batch_indx += 1

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
    coreset(args)