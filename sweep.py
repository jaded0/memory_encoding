prod_supabase_url = "https://djmqhmlaadjjhvwpcxfl.supabase.co"
prod_supabase_anon_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRqbXFobWxhYWRqamh2d3BjeGZsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTY1OTEwNDgsImV4cCI6MjAzMjE2NzA0OH0.tUHEWPT5HebYUpvM4M5smSG1oY_qYeZjO3gG2R-ZWEs"

import os
import requests
import json
from itertools import product

# Define your configuration values here
config_values = {
    "learning_rate": [1e-8],
    "plast_learning_rate": [1e-1,1e-1,1e-1,1e-1,1e-2,1e-2,1e-2,1e-2],
    "update_rule": ['plastic_candidate'],
    "group": ['static_plasticity'],
    "normalize": [False],
    "clip_weights": [0],
    "plast_clip": [1e7],
    "residual_connection": [False],
    "imprint_rate": [1e-6],
    "stochasticity": [1e-40],
    "len_reward_history": [10],
    "delta_rewards": [False],
    "hidden_size": [1024],
    "num_layers": [3],
    "save_frequency": [1000000],
    "n_iters": [1000000000],
    "print_freq": [50],
    "plot_freq": [50],
    "track": [True],
    "dataset": ['4_resequence'],
    "candecay": [0],
    "plast_candecay": [0.999],
    "batch_size": [1]
}

def upload_script_to_database(script, prod_supabase_url, prod_supabase_anon_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {prod_supabase_anon_key}"
    }
    
    body = {"file": script}
    url = f"{prod_supabase_url}/functions/v1/add-run"
    
    response = requests.post(url, headers=headers, data=json.dumps(body))
    
    if response.status_code != 200:
        print(f"An error occurred in the HTTP request: {response.status_code}")
        print(f"Response: {response.text}")
    else:
        print("Script uploaded successfully.")

job_template = """#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8000M   # 8G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=memory_encoding_{learning_rate}_{plast_learning_rate}
#SBATCH --output=./memory_encoding_{learning_rate}_{plast_learning_rate}.out
#SBATCH --mail-user=jaden.lorenc@gmail.com

# some helpful debugging options
set -e
set -u

# nvidia-smi

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate hebby
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export WANDB_MODE=offline

export HF_DATASETS_OFFLINE=1

# run_training.sh

# rm model_data/*

UPDATE_RULE={update_rule}
GROUP={group}
NORMALIZE={normalize}
CLIP_WEIGHTS={clip_weights}
LEARNING_RATE={learning_rate}
PLAST_LEARNING_RATE={plast_learning_rate}
PLAST_CLIP={plast_clip}
RESIDUAL_CONNECTION={residual_connection}
IMPRINT_RATE={imprint_rate}
STOCHASTICITY={stochasticity}
LEN_REWARD_HISTORY={len_reward_history}
DELTA_REWARDS={delta_rewards}
HIDDEN_SIZE={hidden_size}
NUM_LAYERS={num_layers}
SAVE_FREQUENCY={save_frequency}
N_ITERS={n_iters}
PRINT_FREQ={print_freq}
PLOT_FREQ={plot_freq}
TRACK={track}
DATASET={dataset}
CANDECAY={candecay}
PLAST_CANDECAY={plast_candecay}
BATCH_SIZE={batch_size}

python synth_datasets.py

python hebby.py --learning_rate $LEARNING_RATE \\
--group $GROUP \\
--imprint_rate $IMPRINT_RATE \\
--plast_learning_rate $PLAST_LEARNING_RATE \\
--plast_clip $PLAST_CLIP \\
--stochasticity $STOCHASTICITY \\
--len_reward_history $LEN_REWARD_HISTORY \\
--save_frequency $SAVE_FREQUENCY \\
--hidden_size $HIDDEN_SIZE \\
--num_layers $NUM_LAYERS \\
--n_iters $N_ITERS \\
--print_freq $PRINT_FREQ \\
--plot_freq $PLOT_FREQ \\
--update_rule $UPDATE_RULE \\
--normalize $NORMALIZE \\
--clip_weights $CLIP_WEIGHTS \\
--track $TRACK \\
--dataset $DATASET \\
--delta_rewards $DELTA_REWARDS \\
--residual_connection $RESIDUAL_CONNECTION \\
--batch_size $BATCH_SIZE \\
--candecay $CANDECAY \\
--plast_candecay $PLAST_CANDECAY
"""

os.makedirs("jobs", exist_ok=True)

# Get all combinations of hyperparameter values
keys, values = zip(*config_values.items())
for combo in product(*values):
    params = dict(zip(keys, combo))
    job_script = job_template.format(**params)
    job_filename = f"jobs/job_lr_{params['learning_rate']}_plr_{params['plast_learning_rate']}.sh"
    
    with open(job_filename, "w") as f:
        f.write(job_script)
    
    print(f"Generated {job_filename}")
    
    # Upload the generated script
    upload_script_to_database(job_script, prod_supabase_url, prod_supabase_anon_key)
