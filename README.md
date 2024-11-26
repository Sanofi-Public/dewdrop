# DEWDROP-NANOFOLD: Active Learning for Protein Structure Prediction 
This is an implementation of the retrospective experiments in the paper "Active Learning for Protein Structure Prediction". 
* The DEWDROP selection strategy is available in the `alien` module. 
* The Nanofold model is an adaptation of the original [Equifold repository](https://github.com/Genentech/equifold) and is defined in `model.py`. 

The repo can be divided into data pre-processing (`data/`), active learning retrospective experiments (`al_selection/` & `synthetic_example/`), alien implementation of the strategies (`alien/`), and model training/inference scripts. 

## Setup and Usage
### Environment
We use the following GPU-enabled setup with `conda` and `pip`. `conda` is used to manage GPU and bio-conda/conda-forge dependencies. Pip manages the rest.
Note: CUDA_VERSION is the cuda driver version of your pytorch, which should be the same as your compute environment. For example, CUDA_VERSION=118 is version 11.8. 
```
$ sudo apt-get update
$ sudo apt-get install hmmer
$ conda env create -f environment.yml
$ conda activate NanoFold
$ pip install transformers[torch] accelerate
$ pip install ml_collections
$ pip install e3nn
$ pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu${CUDA_VERSION}.html 
$ python -m ipykernel install --user --name NanoFold --display-name "Python (NanoFold)"
```

### Configuration
[config.ini](config.ini) contains high-level configuration for all steps in the model lifecycle, from preprocessing data to benchmarking the final model.
Each section corresponds to a different step in the process.

[models/](models/) contains checkpoints, model weights, and model configuration files that can be used for fine-tuning and inference.

### Data Preprocessing
[data/](data/) contains code for pre-processing .csv and corresponding .pdb or .cif files into the correct format. First, run `python data/equifold_process_input.py` to generate the dataset .pickle file, then run `python data/train_val_test_split.py` for dataset splitting. The `preprocessing` section in `config.ini` should point to the corresponding files and directories.
The output files can later be used for training.

### Run retrospective experiments
[al_selection](al_selection/) contains scripts for different strategies. After processing the data and curated the configuration file, call `python -m al_selection.[name-of-script]` to perform experiments. All the training and validataion statistics are stored under `model_logs/[name-of-configuration]` with the name specified as in the configuration file. 


### End-to-end Training of Nanofold
`python -m run_training` will begin training according to the configuration set in the `training` section. Logging is done with tensorboard by default. Parameters can be set in `config.ini`. 

### Inference
`python -m run_inference` will perform inference on a given set of structures from a csv file. Parameters can be set in `config.ini`. The `n_seeds` parameter in `config.ini` determines how many predictions are made for each input.
[inference_demo.ipynb](inference_demo.ipynb) also walks through the process with visualizations of predicted structures.

Use `python -m run_inference_parallel` to perform inference leveraging multiple GPUs.


***

# Original EquiFold Information
This is the official open-source repository for [EquiFold](https://www.biorxiv.org/content/10.1101/2022.10.07.511322v1) developed by [Prescient Design, a Genentech accelerator.](https://gene.com/prescient)

## Notes
- This light-weight research version of the code was used to produce figures reported in the manuscript (to be updated soon). We plan to release a higher-quality version of the code with additional, user-level features in the future.
- There are known issues occasionally seen in predicted structures, including nonphysical bond geometry and clashes. We are currently developing approaches to minimize these issues for future releases.


## Setup and Usage
### Environment
We used the following GPU-enabled setup with `conda` (originally run in an HPC environment with NVIDIA A100 GPUs).
```
$ conda create -n ef python=3.9 -y
$ conda activate ef
$ conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch -y
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
$ pip install e3nn pytorch-lightning biopython pandas tqdm einops
```

Alternatively, for use without GPUs:
```
conda create -n ef python=3.9 -y
conda activate ef
conda install pytorch=1.12 -c pytorch -y
conda install pyg -c pyg
pip install e3nn pytorch-lightning biopython pandas tqdm einops
```


### Model weights
PyTorch model weights and hyperparameter configs for the models trained on mini-protein and antibody datasets, as described in the manuscript, are stored in the `models` directory.


### Run model predictions
To make predictions using a trained model, users can run the following scripts providing input sequences as a CSV table:

```
# For antibodies
$ python run_inference.py --model ab --model_dir models --seqs tests/data/inference_ab_input.csv --ncpu 1 --out_dir out_tests

# For mini-proteins
$ python run_inference.py --model science --model_dir models --seqs tests/data/inference_science_input.csv --ncpu 1 --out_dir out_tests
```
## Contributing

We welcome contributions. If you would like to submit pull requests, please make sure you base your pull requests on the latest version of the `main` branch. Keep your fork synced by setting its upstream remote to `Genentech/equifold` and running:

```sh
# If your branch only has commits from master but is outdated:

$ git pull --ff-only upstream main


# If your branch is outdated and has diverged from main branch:

$ git pull --rebase upstream main
```

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Citations
If you use the code and/or model, please cite:
```
@article {Lee2022.10.07.511322,
    author = {Lee, Jae Hyeon and Yadollahpour, Payman and Watkins, Andrew and Frey, Nathan C. and Leaver-Fay, Andrew and Ra, Stephen and Cho, Kyunghyun and Gligorijevi{\'c}, Vladimir and Regev, Aviv and Bonneau, Richard},
    title = {EquiFold: Protein Structure Prediction with a Novel Coarse-Grained Structure Representation},
    elocation-id = {2022.10.07.511322},
    year = {2023},
    doi = {10.1101/2022.10.07.511322},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2023/01/02/2022.10.07.511322},
    eprint = {https://www.biorxiv.org/content/early/2023/01/02/2022.10.07.511322.full.pdf},
    journal = {bioRxiv}
}
```