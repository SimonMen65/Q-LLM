# datasets='narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p' # long bench
datasets='gov_report'
world_size=2

config=llama3-qllm-repr4-l1k-bs128-topk8-w4
config_path=config/$config.yaml
output_dir_path=result/longbench/$config

# make prediction
bash scripts/multiprocessing-benchmark.sh \
    --world_size $world_size \
    --config_path $config_path \
    --output_dir_path ${output_dir_path} \
    --datasets $datasets

# merge multi-process results
python benchmark/merge.py \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size}

# evaluation
python benchmark/eval.py \
    --dir_path ${output_dir_path}