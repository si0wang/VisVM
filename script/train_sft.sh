#!/bin/bash

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./deepspeed_zero3.yaml --num_processes=8 \
    ./sft_training.py \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --output_dir ./llava_1.6_mistal_sft \
    --bf16 \
    --save_steps 50 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --log_level info \
    --logging_steps  10 \
    --logging_strategy steps \
    --gradient_checkpointing