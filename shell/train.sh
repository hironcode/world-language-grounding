#!/bin/bash

#SBATCH --job-name=MELD
#SBATCH --partition=gpu2
#SBATCH --mem=240Gb
#SBATCH --nodes=4
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

# output file
#SBATCH --output=./logs_slurm/train_mp_meld/output/R-%x.%j.out
#SBATCH --error=./logs_slurm/train_mp_meld/error/R-%x.%j.err

# these echos is going to be printed on the .out file.

echo "===================================="
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`."
echo "===================================="

module load python39
module load cuda11.0/toolkit/11.0.3

# ensure environment is ready for Tensorflow GPU
echo "Cheking for NVIDIA devices:"
nvidia-smi

# activate environment (before)
source /home/nfx18/python-venv/bin/activate
echo "After loading python39, current working python is `which python`"

cd /home/nfx18/reu2024/FARSER


# Python Code
python3 -u <<EOF

import os
print(os.getcwd())

import torch
from torch import optim

try:
    from modeling.model import FarcerModel, ParamsFarcer
    from modeling import util
    from src import features, config
except:
    from model import FarcerModel, ParamsFarcer
    import util
    from ..src import features, config
from modeling.train import train_epochs_modality_projector

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load model
params = ParamsFarcer()

if torch.cuda.is_available():
    print(f"AVILABLE CUDA DEVICE COUNT: {torch.cuda.device_count()}")

params.lm_quantize = None
params.num_mp_layers = 1
params.vm_only_use_cls_token = True

model = FarcerModel(params)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
EPOCHS = 20
BATCH_SIZE = 1
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

message = "Training conditions: 7 shots, 1 batch size, 20 epochs, MELD dataset, early stopping with f1 metric, radom 10 prompts, eos token added to labels, linear layers=3, traning data=train + dev. I changed the learning rate back to 0.0003 to see the performnce."

train_epochs_modality_projector(model, optimizer, scheduler, batch_size=BATCH_SIZE, num_epochs=EPOCHS, dataset_type="MELD", early_stopping=True, prompt_list=True, message=message)
print("Training complete")

exit()

EOF

module unload python39
module unload cuda11.0/toolkit/11.0.3

# exit()

