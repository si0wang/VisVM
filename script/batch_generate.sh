#!/bin/bash

model_id="llava-hf/llava-v1.6-mistral-7b-hf"
output_prefix="./output_files/pretrain_value_batch_generate_llava1_6_mistral_7b_res_"
final_file_prefix="./pretrain_value_batch_generate_llava1_6_mistral_7b_res"
batch_size=16
num_chunks=8

for i in {0..8}; do
     python batch_generate.py \
        --model_id $model_id \
        --output_file ${output_prefix}$((i+1)).jsonl \
        --per_gpu_batch_size $batch_size \
        --num-chunks $num_chunks \
        --chunk-idx $((i)) \
        --gpu-id $((i)) &
done

wait

cat ${output_prefix}*.jsonl > ${final_file_prefix}.jsonl