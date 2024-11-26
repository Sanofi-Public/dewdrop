"""
Run one or multiple dewdrop experiments on multiple GPUs
--- 
1. Specify the parameter for each experiments in the config_dewdrop_ab.ini file under the [dewdrop] section. 
2. Run this script to run parallel processes that will perfrom different trials of dewdrop retroexperiment. 
"""
import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import json
import os
import tqdm
import pickle
from pathlib import Path
import shutil
from multiprocessing import Process
from ast import literal_eval
import numpy as np

if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())

    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('Base CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    # Read training configuration from json 
    training_config_path = CONFIG["training"]["training_config_path"]
    initial_weights = CONFIG["training"]["initial_weights"]
    version_id = CONFIG["training"]["version_id"]

    with open(training_config_path, "r") as f:
        training_config = json.load(f)

    # Read how many expt to run
    experiments = literal_eval(CONFIG['dewdrop']['experiments'])
    device_list = literal_eval(CONFIG['dewdrop']['devices_list'])
    batchsize_list = literal_eval(CONFIG['dewdrop']['batchsize_list'])
    ensemblesize_list = literal_eval(CONFIG['dewdrop']['ensemblesize_list'])
    modified_args = []

    print("Experiments to run: ", experiments)

    for i, exp in enumerate(experiments): 
        # flush the configuration file 
        NEW_CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        NEW_CONFIG.read_dict(CONFIG)

        # create modified copy of the training_config
        expt_name = f"dewdrop_{exp}"
        NEW_CONFIG["training"]["run_name"] = expt_name
        # CONFIG["training"]["output_dir"] = os.path.join(CONFIG["training"]["output_dir"], expt_name)
        training_config_fp = os.path.join(os.path.dirname(CONFIG['training']['training_config_path']), f"training_{exp}.json")
        weight_fp = os.path.join(os.path.dirname(CONFIG['training']['initial_weights']), f"weights_{exp}_{version_id}.pt")

        # Change training parameters
        new_training_config = copy.deepcopy(training_config)
        new_training_config["train_batch_size"] = batchsize_list[i]
        new_training_config["ensemble_size"] = ensemblesize_list[i]
        new_training_config["pca_components"] = ensemblesize_list[i] # ensemble_size == pca_components 

        # Save a new copy of the weight 
        shutil.copy2(CONFIG['training']['initial_weights'], weight_fp)

        # Save a copy of hte training config file 
        with open(training_config_fp, 'w') as file: 
            json.dump(new_training_config, file)

        # Change to the new filepaths
        NEW_CONFIG["training"]["training_config_path"] = training_config_fp
        NEW_CONFIG["training"]["initial_weights"] = weight_fp
        NEW_CONFIG["training"]["devices"] = str(device_list[i])

        expt_config_fp = f"config_dewdrop_{exp}.ini"
        with open(expt_config_fp, "w") as configfile: 
            NEW_CONFIG.write(configfile)
        new_args = copy.deepcopy(args)
        new_args["config"] = expt_config_fp
        modified_args.append(new_args) 