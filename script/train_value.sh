#!/bin/bash

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./deepspeed_zero3.yaml --num_processes=8 \
    ./value_training.py \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --output_dir ./visvm-ckpt \
    --bf16 \
    --save_steps 200 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --log_level info \
    --logging_steps  50 \
    --logging_strategy steps \
    --gradient_checkpointing