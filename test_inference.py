import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gzip
import json
import os
import subprocess
import time
from multiprocessing import Pool
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning  as pl 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Change relative imports 
from data.equifold_process_input import data_process
from model_utils.models import NN
from openfold_light.residue_constants import restype_3to1

from utils.refine import refine
from utils.sequence_checks import number_sequences
from utils.utils import compute_prediction_error, to_atom37
from utils.utils_data import collate_fn, x_to_pdb

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aggregate_predictions(predictions):
    aggregated = {
        "predictions": [],
        "additional_info": []
    }
    for pred in predictions:
        # Collect and concatenate values from all predictions
        aggregated["predictions"].append(pred["prediction"].cpu())
        aggregated["additional_info"].extend(pred["additional_info"])

    # Further aggregation if needed (e.g., stacking tensors)
    aggregated["predictions"] = torch.cat(aggregated["predictions"], dim=0)
    return aggregated


if __name__ == "__main__":
    
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./configs/config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())

    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    t_0 = time.time()

    # fined-tuned model
    model_path = CONFIG["inference"]["model_ckpt"]
    config_path = CONFIG["inference"]["config_path"]
    with open(config_path, "r") as f:
        config = json.load(f)

    model = NN(**config)
    checkpoint = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Finish loading\n")

    # load data
    df = pd.read_csv(CONFIG["inference"]["sequence_path"])
    uids = df["uid"].tolist()
    if CONFIG["inference"]["model_type"] == "ab":
        seqs1 = df["heavy"].tolist()
        seqs2 = df["light"].tolist()
    else:
        seqs1 = df["seq"].tolist()
        seqs2 = [None] * len(seqs1)

    # Add chain_id into data object
    chain_id = df["chain_id"].tolist()

    # prepare data structures using multiproc
    jobs = list(zip(uids, seqs1, seqs2, chain_id))
    print(f"Total number of sequences: {len(jobs)}")
    
    if CONFIG["inference"]["ncpu"] != "":
        ncpu = CONFIG.getint("inference", "ncpu")
    else:
        ncpu = os.cpu_count()
    
    def init_worker(config):
        global processor
        processor = data_process(config)

    def process_one_worker(job):
        # Access the worker-local processor
        return processor.process_one(job)

    with Pool(ncpu, initializer=init_worker, initargs=(CONFIG,)) as p:
        dataset = list(tqdm(p.imap_unordered(process_one_worker, jobs), total=len(jobs)))


    # run (multi-gpu) inference and save
    loader = DataLoader(
        dataset,
        batch_size=32,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # (TODO) subject to change 
    # GPUs = 1
    GPUs = torch.cuda.device_count()
    print("Trainer is using: ", GPUs, "GPUs.")
    predicter = pl.Trainer(
        strategy="ddp",
        accelerator="gpu",
        devices=GPUs
    )

    t_1 = time.time()

    # os.makedirs(CONFIG["inference"]["output_dir"], exist_ok=True)
    # with torch.no_grad():
    #     pl.seed_everything(0)
    #     preds = predicter.predict(model, dataloaders=loader)
        
        # pdb.set_trace()

    from torch.distributed import is_initialized

    # Function to gather predictions across all GPUs
    def gather_predictions(preds):
        if is_initialized():  # Check if running in a distributed setting
            import torch.distributed as dist
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # Gather all predictions into a list
            gathered_preds = [None] * world_size
            dist.all_gather_object(gathered_preds, preds)

            # Only rank 0 consolidates the results
            if rank == 0:
                consolidated_preds = []
                for pred in gathered_preds:
                    consolidated_preds.extend(pred)
                return consolidated_preds
            else:
                return None
        else:
            return preds  # No distribution, return as is

    preds = predicter.predict(model, dataloaders=loader)

    # Gather predictions from all GPUs
    consolidated_preds = gather_predictions(preds)

    if consolidated_preds is not None:  # Only save on rank 0
        print("Size of the prediction: ", len(consolidated_preds))
        output_path = os.path.join(CONFIG["inference"]["output_dir"], "predictions.pt")
        torch.save(consolidated_preds, output_path)
                
        t_2 = time.time()
        print(f"Data processing done in {round(t_1-t_0)}s.")
        print(f"Inference done in {round(t_2-t_1)}s.")
        print(f"Average inference time is {round(t_2-t_1)/len(jobs)}s.")
