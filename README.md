<div align="center">

# CryoEM Denoising

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>
<div align='left'>
</div>

## Description

Main repository for MultiNoise2Noise for CryoEM project. It contains source code for experiments and various algorithms for processing of cryoEM data.

## Installation
In `configs_train/local` create file `default.yaml` and set it:
```
user_name: your-user-name
storage_dir: /path/to/your/data/directory
```

Create conda environment using
```
python -m venv multinoise2noise
source multinoise2noise/bin/activate
```

Install dependencies using pip
```
pip install -r requirements.txt
```

## Project structure

### Overall structure
```
├── configs_extract               # configs for extraction of particles from micrographs
├── configs_motion_correction     # configs for motion correction of micrographs
├── configs_train                 # configs for training
├── notebooks                     # directory for Notebooks
├── scripts                       # package for bash scripts
└── source                        # source code package
│   ├── _extraction               # lightning model and components for extraction, fourier cropping and normalization of images.
│   ├── _motion_correction        # lightning model and components for Motion correction of micrograph stacks. It provides outputs not summed to singular frame, allowing application of MultiNoise2Noise.
│   ├── _noise2noise              # lightning model and components for Noise2Noise family models
│   ├── datasets                  # datamodules and datasets
│   ├── loss_functions            # loss functions that can be used for training 
│   ├── models                    # neural networks, such as UNets
│   ├── reconstruction            # all modules and components used in process of reconstruction of 3D models with particles. Contains methods such as CTF Correction, Dose Weighting, B-Factor sharpening and more.
│   ├── utils                     # utility, like logging etc
│   │     
│   ├── denoise.py                # script for denoising micrographs
│   ├── evaluate_tem.py           # script for evaluating methods
│   └── train.py                  # scripts for training methods
├── requirements.txt              # file for pip environment
└── README.md                     # this README
```

### Training configs structure
```
├── configs_train
│   ├── callbacks           # Lightning callbacks.
│   ├── core                # Metadata such as project-name, seed etc.
│   ├── datamodule          # Overall data setup.
│   ├── debug               # For debugging purposes, such as overfitting to one batch.
│   ├── experiment          # Ready-to-use experiments.
│   ├── extras              # Additional options such as asking for tags etc.
│   ├── hparams_search      # Setup for Optuna hyperparameter search.
│   ├── hydra               # Setup for Hydra. Leave it as it is.
│   ├── lightning_model     # Lightning models like Noise2NoiseLightningModel.
│   ├── local               # Local setup for particular user and machine. Don't push it on Github.
│   ├── logger              # Different loggers.
│   ├── model               # Models (like UNets etc).
│   ├── paths               # Paths to core directories, logs etc.
│   ├── trainer             # Trainer configuration, like number of GPUs, number of epochs etc.
│   └── train.yaml          # Main config file.
```

## How to train model

Train model with default configuration

```bash
# train on CPU
python3 -m source.train trainer=cpu

# train on GPU
python3 -m source.train trainer=gpu
```

Train model with chosen experiment configuration from [configs_train/experiment/](configs_train/experiment/). Name of configuration is full path starting after 'experiment', up to yaml file.

```
python3 -m source.train experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```
python3 -m source.train trainer.max_epochs=20 datamodule.batch_size=64
```

## How to denoise micrographs

In `configs_denoise` create your configuration file. Check out the existing example. Then you can run the command using
```
python3 -m source.denoise experiment=experiment_name.yaml
```
