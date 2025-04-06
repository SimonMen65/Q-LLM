# datasets='narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p' # long bench
datasets='gov_report'
world_size=1

config=llama3-qllm
config_path=config/$config.yaml
output_dir_path=results/longbench/$config

# make prediction
bash scripts/multiprocessing-benchmark.sh \
    --world_size $world_size \
    --config_path $config_path \
    --output_dir_path ${output_dir_path} \
    --datasets $datasets

# # merge multi-process results
# python benchmark/merge.py \
#     --output_dir_path ${output_dir_path} \
#     --datasets ${datasets} \
#     --world_size ${world_size}

# # evaluation
# python benchmark/eval.py \
#     --dir_path ${output_dir_path}