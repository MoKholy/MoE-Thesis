#!/bin/bash
#SEEDS=(10, 20, 30, 40, 50)
SEEDS=(10)
for seed in ${SEEDS[@]}; do
    python main.py \
       --seed $seed \
       --dataset_name "cleaned_v2_dataset.csv" \
       --dataset_path "data/processed/" \
       --batch_size 128 \
       --hidden_dim 128 \
       --dropout 0.3 \
       --activation_fn "relu" \
       --num_epochs 100 \
       --learning_rate 0.001 \
       --weight_decay 0.0001 \
       --expert_loss_fn "focal_loss" \
       --optimizer "adam" \
       --betas 0.9 0.999 \
       --scheduler "lr_step" \
       --scheduler_step_size 50 \
       --gamma   0.1 \
       --save_dir "runs/" \
       --model_name "test" 
done