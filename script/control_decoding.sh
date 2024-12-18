#!/bin/bash

output_prefix="./output_files/value_net_decoding_results_"
final_file_prefix="./value_net_decoding_results"
num_chunks=8

for i in {0..7}; do
     python control_decoding.py \
        --output_file ${output_prefix}$((i+1)).jsonl \
        --step_size 1 \
        --num-chunks $num_chunks \
        --chunk-idx $((i)) \
        --gpu-id $((i)) &
done

wait

cat ${output_prefix}*.jsonl > ${final_file_prefix}.jsonl