#!/bin/bash

# 配置
datasets='gov_report'
world_size=1
config=llama3-qllm
config_path=config/$config.yaml
output_dir_path=results/longbench/$config

mkdir -p ${output_dir_path}

trap 'kill $(jobs -p)' SIGINT

echo "============ START PRED ============"

for ((rank=0; rank < $world_size; ++rank))
do
    CUDA_VISIBLE_DEVICES=${rank} python benchmark/pred.py \
    --num_samples 1 \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank ${rank} &
    echo "worker $rank started"
done

wait

echo "============ START MERGE ============"

# python benchmark/merge.py \
#     --output_dir_path ${output_dir_path} \
#     --datasets ${datasets} \
#     --world_size ${world_size}

echo "============ START EVAL ============"

python benchmark/eval.py --dir_path ${output_dir_path}
