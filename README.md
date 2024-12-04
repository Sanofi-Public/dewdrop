# DEWDROP-NANOFOLD: Active Learning for Protein Structure Prediction 
This is an implementation of the retrospective experiments in the paper "Active Learning for Protein Structure Prediction". 
* The DEWDROP selection strategy is available in the `alien` module, which is a modified version of [ALIEN](https://github.com/Sanofi-Public/Alien). 
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

### Model Weights and Configurations
[configs/](configs/) contains .ini files which are high-level configuration for all steps in the model lifecycle, from preprocessing data to benchmarking the final model.
Each section corresponds to a different step in the process.

[models/](models/) contains checkpoints, model weights, and model configuration files that can be used for fine-tuning and inference.
There are serveral model weights and configurations: the original Equifold version under `ab/`, the recycling version under `models_with_recycling/`, and nanobody finetuned version `nanobody_finetune_200`. For running retrospective experiment, we start with the original weight and configuration. 

### Data Preprocessing
[data/](data/) contains code for pre-processing .csv and corresponding .pdb or .cif files into the correct format. 
If you want to run any training, you need to run `python data/prepare_pdb_1700.py` to obtain the .pdb files for all the names; otherwise, just having the sequences in a .csv file is fine. 
After that, run `python data/equifold_process_input.py` to generate the dataset .pickle file, then run `python data/train_val_test_split.py` for dataset splitting. The `preprocessing` section in `config.ini` should point to the corresponding files and directories.
The output files can later be used for training.


### Run retrospective experiments
[al_selection](al_selection/) contains scripts for different strategies. After processing the data and curated the configuration file, call `python -m al_selection.[name-of-script]` to perform experiments. All the training and validataion statistics are stored under `model_logs/[name-of-configuration]` with the name specified as in the configuration file. 

### End-to-end Training of Nanofold
`python -m run_training` will begin training according to the configuration set in the `training` section. Logging is done with tensorboard by default. Parameters can be set in `config.ini`. 

### Inference
`python -m run_inference` will perform inference on a given set of structures from a csv file. Parameters can be set in `config.ini`. The `n_seeds` parameter in `config.ini` determines how many predictions are made for each input.
[inference_demo.ipynb](inference_demo.ipynb) also walks through the process with visualizations of predicted structures.

Use `python -m run_inference_parallel` to perform inference leveraging multiple GPUs.

## Citations
```
@article{Xue2024,
  author       = {Xue, Zexin and Bailey, Michael and Gupta, Abhinav and Li, Ruijiang and Corrochano-Navarro, Alejandro and Li, Sizhen and Kogler-Anele, Lorenzo and Yu, Qui and Bar-Joseph, Ziv and Jager, Sven},
  title        = {Active Learning for Protein Structure Prediction},
  note         = {Under review},
  year         = {2024},
  institution  = {R\&D Data \& Computational Science, Sanofi, Cambridge, MA, United States}
}

```
