#!/bin/bash

# Arrays of epsilons and datasets
epsilons=(0.1 1 10 25 50)
datasets=("rte")
seeds=(7 42 124)
# Base command components
model_name="microsoft/deberta-v3-base"
max_seq_length=256
train_batch_size=64
learning_rate=2e-5
num_epochs=1
output_dir_base="./checkpoints/"

# Log directory
log_dir="./logs_MST"
mkdir -p "$log_dir"  # Create logs directory if it doesn't exist

# Loop through each combination of epsilon and dataset
for epsilon in "${epsilons[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            train_file="datasets_MST_MVC_epsilon_${epsilon}/${dataset}_train.csv"
            validation_file="datasets_MST_MVC_epsilon_${epsilon}/${dataset}_validation.csv"
            output_dir="${output_dir_base}_${dataset}_epsilon_${epsilon}_seed_${seed}"
            log_file="${log_dir}/${dataset}_epsilon_${epsilon}_seed_${seed}.txt"

            # Run the Python command and redirect output to log file
            echo "Running for epsilon=${epsilon}, dataset=${dataset}. Logs will be saved to ${log_file}"
            python run_glue.py \
                --model_name_or_path "$model_name" \
                --do_train \
                --do_eval \
                --max_seq_length "$max_seq_length" \
                --per_device_train_batch_size "$train_batch_size" \
                --learning_rate "$learning_rate" \
                --num_train_epochs "$num_epochs" \
                --output_dir "$output_dir" \
                --overwrite_output_dir \
                --seed "$seed" \
                --train_file "$train_file" \
                --validation_file "$validation_file" \
                > "$log_file" 2>&1

            # Cleanup: Delete the output directory after the run is complete
            if [ -d "$output_dir" ]; then
                echo "Deleting output directory: $output_dir"
                rm -rf "$output_dir"
            fi
        done
    done
done
