## Self-Supervised Vit for Redshift Estimation
This repo is for Zhuoting Xie's project assignment to EECE570.

## Installation
To install the python enviromnent, make sure you have conda installed. \
Run `conda env create -f environment.yml` to create the environment. \
Then activate use `conda activate redshift_env`.

## Data
Please download image data from `https://drive.google.com/file/d/1o9pxvoQjQPbY5IOipvG5ccf7Wq2f2J1c/view?usp=sharing` . Unzip and place all fits files under `data/pdr3/input/input_fits/`.

## Running
Parameters can be set in `configs/redshift.yaml`.

To perform pretraining
- set `trainer_mode` to `pre_training`.

To perform supervised training, 
- set `trainer_mode` to `redshift_train`.
- set `pretrained_model_dir` to the model directory for pretraining (e.g. `20230416-180501-pre_training`).
- set `pretrained_model_fname` to the checkpoint file name you want to use (e.g. `model-ep100-it129.pth`).

To perform testing:
- set `trainer_mode` to `test`
- set `best_model_log_dir` to the model directory for supervised training (e.g. `20230419-103925-redshift_train`).
- set `pretrained_model_fname` to the checkpoint file name you want to use (e.g. `best_model.pth`).
