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
import lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.equifold_process_input import data_process
from model_utils.models import NN
from openfold_light.residue_constants import restype_3to1
from utils.refine import refine
from utils.sequence_checks import number_sequences
from utils.utils import compute_prediction_error, to_atom37
from utils.utils_data import collate_fn, x_to_pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save(pred_filename, relaxed_filename, check_for_strained_bonds=True, n_threads=-1):
    success = refine(
        pred_filename,
        relaxed_filename,
        check_for_strained_bonds=check_for_strained_bonds,
        n_threads=n_threads,
    )
    if success:
        return
    else:
        success = refine(
            pred_filename,
            relaxed_filename,
            check_for_strained_bonds=check_for_strained_bonds,
            n_threads=n_threads,
        )
        if success:
            return

    if not success:
        print(f"FAILED TO REFINE {pred_filename}.\n", flush=True)


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
    model = model.to(device)
    model.eval()
    print("Device: ", model.device)
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

    with Pool(ncpu) as p:
        dataset = list(tqdm(p.imap_unordered(data_process(CONFIG).process_one, jobs), total=len(jobs)))
        p.close()
        p.join()

    # run inference and save
    loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    t_1 = time.time()

    os.makedirs(CONFIG["inference"]["output_dir"], exist_ok=True)
    with torch.no_grad():
        for data in tqdm(loader):
            # pylint: disable=not-callable
            data = data.to(device)
            x_preds_atom37 = []
            if CONFIG.getint("inference", "n_seeds") == 1:
                pl.seed_everything(0)
                results_dict = model(
                    data, compute_loss=False, return_struct=True, set_RT_to_ground_truth=False
                )
                x_pred = results_dict["x_pred"][0][-1]
                # write pred
                input_name = f"{CONFIG['inference']['output_dir']}/{data[0].uid}_{data[0].chain_id}_rank0_unrefined.pdb"
                with gzip.open(input_name + ".gz", "wb") as f:
                    f.write(
                        x_to_pdb(
                            x_pred.cpu(),
                            data[0]["dst_resnum"],
                            data[0]["dst_resname"],
                            data[0]["dst_atom"],
                        ).encode()
                    )
                relaxed_name = f"{CONFIG['inference']['output_dir']}/{data[0].uid}_{data[0].chain_id}_final_model.pdb"
                if os.path.isfile(input_name):
                    os.remove(input_name)
                subprocess.call(["gunzip", input_name + ".gz"])
                if CONFIG.getboolean("inference", "refine"):
                    save(input_name, relaxed_name)
                else:
                    shutil.copy(input_name, relaxed_name)
            else:
                for seed in range(CONFIG.getint("inference", "n_seeds")):
                    # Seeding
                    pl.seed_everything(seed)
                    results_dict = model(
                        data, compute_loss=False, return_struct=True, set_RT_to_ground_truth=False
                    )

                    # get pred
                    x_pred = results_dict["x_pred"][0][-1]
                    x_preds_atom37.append(
                        torch.squeeze(
                            to_atom37(
                                x_pred,
                                data[0]["dst_resnum"],
                                data[0]["dst_atom"],
                                data[0]["dst_resname"],
                            )[0]
                        )
                    )

                # Get sequence names
                res, ind = np.unique(data[0]["dst_resnum"].cpu().numpy(), return_index=True)
                seq_full_name = data[0]["dst_resname"][np.sort(ind)]
                seq_short_name = [
                    restype_3to1.get(seq_full_name[i], "X") for i in range(len(seq_full_name))
                ]
                seq_short_name = "".join(seq_short_name)
                numbered_sequences = number_sequences({"H": seq_short_name}, allowed_species=None)
                numbered_sequences["L"] = []

                obj = compute_prediction_error(
                    numbered_sequences,
                    x_preds_atom37,
                    refine=CONFIG.getboolean("inference", "refine"),
                )
                # ADDED: catch save error 
                try: 
                    obj.save_all(
                        uid=data[0]["uid"] + "_" + data[0]["chain_id"],
                        dirname=CONFIG["inference"]["output_dir"],
                    )
                except KeyError: 
                    print("Skip saving for invalid prediction. ")

    t_2 = time.time()
    print(f"Data processing done in {round(t_1-t_0)}s.")
    print(f"Inference done in {round(t_2-t_1)}s.")
    print(f"Average inference time is {round(t_2-t_1)/len(jobs)}s.")
