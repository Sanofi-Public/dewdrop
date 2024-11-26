import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gzip
import json
import os
import subprocess
import time
import multiprocessing as mp
import shutil
import pdb

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils_parallel import DataParallel, compute_prediction_error

from data.equifold_process_input import data_process
from model_utils.models import NN
from openfold_light.residue_constants import restype_3to1
from utils.refine import refine
from utils.sequence_checks import number_sequences
from utils.utils import to_atom37, add_errors_as_bfactors
from utils.utils_data import collate_fn, x_to_pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def save(job):
    uid, chain, gpu_id, header, dirname= job
    
    pred_filename = f"{dirname}/{uid}_{chain}_rank0_unrefined.pdb"
    relaxed_filename = f"{dirname}/{uid}_{chain}_final_model.pdb"
    error_filename = f"{dirname}/{uid}_{chain}_error_estimates.npy"

    success = refine(
        pred_filename,
        relaxed_filename,
        check_for_strained_bonds=True,
        n_threads=-1,
        platform="CUDA" if device=="cuda:0" else None, 
        CudaDeviceIndex=gpu_id
    )
    if success:
        pass
    else:
        success = refine(
            pred_filename,
            relaxed_filename,
            check_for_strained_bonds=True,
            n_threads=-1,
            platform="CUDA" if device=="cuda:0" else None, 
            CudaDeviceIndex=gpu_id
        )

    if not success:
        print(f"FAILED TO REFINE {pred_filename}.\n", flush=True)
        shutil.copy(os.path.join(dirname, f"{uid}_{chain}_rank0_unrefined.pdb"), relaxed_filename)

    if os.path.isfile(error_filename):
        error_estimates = np.load(error_filename)
        add_errors_as_bfactors(
                relaxed_filename, np.sqrt(error_estimates), header=[header]
            )
    return

class model_wrapper(torch.nn.Module):
    def __init__(self, model, compute_loss=False, return_struct=True, set_RT_to_ground_truth=False):
        super().__init__()
        self.model = model
        self.compute_loss = compute_loss
        self.return_struct = return_struct
        self.set_RT_to_ground_truth = set_RT_to_ground_truth
        self.DataParallelMode = False
    
    def forward(self, data):
        if self.DataParallelMode:
            output=self.model([data], self.compute_loss, self.return_struct, self.set_RT_to_ground_truth)
        else:
            output=[self.model(data, self.compute_loss, self.return_struct, self.set_RT_to_ground_truth)]
        return output

class process_output():
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG

    def single(self, job):
        x_pred, data = job
        input_name = f"{self.CONFIG['inference']['output_dir']}/{data.uid}_{data.chain_id}_rank0_unrefined.pdb"
        with gzip.open(input_name + ".gz", "wb") as f:
            f.write(
                x_to_pdb(
                    x_pred,
                    data["dst_resnum"],
                    data["dst_resname"],
                    data["dst_atom"],
                ).encode()
            )
        if os.path.isfile(input_name):
            os.remove(input_name)
        subprocess.call(["gunzip", '-f', input_name + ".gz"])

        final_name = f"{self.CONFIG['inference']['output_dir']}/{data.uid}_{data.chain_id}_final_model.pdb"
        shutil.copy(input_name, final_name)

    def multiple(self, job):
        x_preds_atom37, data = job
        # Get sequence names
        res, ind = np.unique(data["dst_resnum"].numpy(), return_index=True)
        seq_full_name = data["dst_resname"][np.sort(ind)]
        seq_short_name = [
            restype_3to1.get(seq_full_name[i], "X") for i in range(len(seq_full_name))
        ]
        seq_short_name = "".join(seq_short_name)
        numbered_sequences = number_sequences({"H": seq_short_name}, allowed_species=None)
        numbered_sequences["L"] = []
        
        obj = compute_prediction_error(
            numbered_sequences,
            x_preds_atom37,
            refine=self.CONFIG.getboolean("inference", "refine")
        )
        obj.save_all(
            uid=data["uid"] + "_" + data["chain_id"],
            dirname=self.CONFIG["inference"]["output_dir"],
        )

        dirname = self.CONFIG['inference']['output_dir']
        pred_filename = f"{dirname}/{data.uid}_{data.chain_id}_rank0_unrefined.pdb"
        final_name = f"{dirname}/{data.uid}_{data.chain_id}_final_model.pdb"
        shutil.copy(pred_filename, final_name)

if __name__ == "__main__":

    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
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

    with mp.Pool(ncpu) as p:
        dataset = list(tqdm(p.imap_unordered(data_process(CONFIG).process_one, jobs), total=len(jobs)))
        p.close()
        p.join()

    # run inference and save
    model_parallel = model_wrapper(model, compute_loss=False, return_struct=True, set_RT_to_ground_truth=False)
    model_parallel.to(device)
    batch_size = 1
    gpu_queue = [None] * len(uids)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_parallel.DataParallelMode = True
        model_parallel = DataParallel(model_parallel)
        batch_size = torch.cuda.device_count()
        print('Done with creating model replicates!')
        gpu_queue = [i % batch_size for i in range(len(uids))]
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    os.makedirs(CONFIG["inference"]["output_dir"], exist_ok=True)
    
    t_1 = time.time() #Do not count the time needed to replicate the model on different devices because it is done only once
    print(f"Data processing and model initialization done in {round(t_1-t_0)}s.")
    
    x_pred_total = []
    x_preds_atom37_total = []
    data_total = []
    
    with torch.no_grad():
        for data in tqdm(loader):
            # pylint: disable=not-callable
            
            data_total = data_total + [x.clone() for x in data]
            data.to(device)
            
            if CONFIG.getint("inference", "n_seeds") == 1:
                pl.seed_everything(0)
                results_dict = model_parallel(data)
                
                x_pred_total = x_pred_total + [x["x_pred"][0][-1].cpu() for x in results_dict]

            else:
                x_preds_atom37 = [[] for i in range(len(data))]
                for seed in range(CONFIG.getint("inference", "n_seeds")):
                    # Seeding
                    pl.seed_everything(seed)
                    results_dict = model_parallel(data)

                    # get pred
                    for i in range(len(data)):
                        x_pred = results_dict[i]["x_pred"][0][-1].cpu()
                        x_preds_atom37[i].append(
                            torch.squeeze(
                                to_atom37(
                                    x_pred,
                                    data[i]["dst_resnum"],
                                    data[i]["dst_atom"],
                                    data[i]["dst_resname"],
                                )[0]
                            )
                        )
                
                x_preds_atom37_total = x_preds_atom37_total + x_preds_atom37
    
    print('Let us save all the result!')
    
    process_obj = process_output(CONFIG)
    
    if CONFIG.getint("inference", "n_seeds") == 1:
        jobs = list(zip(x_pred_total, data_total))

        with mp.get_context('fork').Pool(ncpu) as p:
            list(tqdm(p.imap_unordered(process_obj.single, jobs), total=len(jobs)))
            p.close()
            p.join()
    else:
        jobs = list(zip(x_preds_atom37_total, data_total))

        with mp.get_context('fork').Pool(ncpu) as p:
            list(tqdm(p.imap_unordered(process_obj.multiple, jobs), total=len(jobs)))
            p.close()
            p.join()
    
    if CONFIG.getboolean("inference", "refine"):
        t_r_1 = time.time()
        print('Starting refinement')
        header=[""] * len(uids)
        dirname = [CONFIG['inference']['output_dir']] * len(uids)
        jobs = list(zip(uids, chain_id, gpu_queue, header, dirname))

        with mp.get_context('spawn').Pool(ncpu) as p:
            list(tqdm(p.imap_unordered(save, jobs), total=len(jobs)))
            p.close()
            p.join()

        t_r_2 = time.time()
        print(f"Refinement done in {round(t_r_2-t_r_1)}s.")
    else: 
        print("No refinement will be done!")

    t_2 = time.time()
    
    print(f"Inference done in {round(t_2-t_1)}s.")
    print(f"Average inference time is {round(t_2-t_1)/len(uids)}s.")
