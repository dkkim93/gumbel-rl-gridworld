#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

# Begin experiment
for SEED in {1..1}
do
    python3 main.py \
    --env-name "Gridworld-v0" \
    --ep-max-timesteps 10 \
    --row 3 \
    --col 11 \
    --prefix ""
done
