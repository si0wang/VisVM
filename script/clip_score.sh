#!/bin/bash

clip_id="openai/clip-vit-large-patch14-336"
input_prefix="./output_files/pretrain_value_batch_generate_llava1_6_mistral_7b_res_"
output_prefix="./output_files/pretrain_clip_score_"
final_file_prefix="./pretrain_clip_score"

for i in {0..7}; do
     CUDA_VISIBLE_DEVICES=i python generate_clip_score.py \
        --clip_id $clip_id \
        --data_pth ${input_prefix}$((i+1)).jsonl \
        --output_file ${output_prefix}$((i+1)).jsonl \
        --gpu-id $((i)) &
done

wait

cat ${output_prefix}*.jsonl > ${final_file_prefix}.jsonl