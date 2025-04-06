import os
import argparse
from pred import load_infinite_bench
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    args = parser.parse_args()
    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    datasets_list = [s.strip() for s in datasets_list]
    args.datasets = datasets_list
    return args

if __name__ == "__main__":
    args = parse_args()
    for dataset in args.datasets:

        out_path = os.path.join(
            args.output_dir_path,
            f"{dataset}.jsonl"
        )

        lines = []

        if args.world_size == 1:
            # world_size=1 直接读取 out_path 自己
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    lines = [l.strip() for l in f.readlines()]
            print(f"single world\n")
        else:
            # world_size>1 拼接多个 _0 _1 _2
            for rank in range(args.world_size):
                file_path = out_path + f"_{rank}"
                if not os.path.exists(file_path):
                    continue
                with open(file_path, "r") as f:
                    lines += f.readlines()
            lines = [l.strip() for l in lines]

        f = open(out_path, "w+")
        f.write(
            "\n".join(lines)
        )
        f.close()
        
        if dataset in set([
            "kv_retrieval", "passkey", "number_string", "code_run", "code_debug", "longdialogue_qa_eng", "longbook_qa_eng", "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_chn", "math_find", "math_calc"
        ]):
            path = "benchmark/data/infinite-bench"
            data = load_infinite_bench(path, dataset)
        elif dataset in set([
            "kv_retrieval_32k", "kv_retrieval_64k", "kv_retrieval_128k", "kv_retrieval_256k", "kv_retrieval_512k", "kv_retrieval_768k", "kv_retrieval_1024k", 
        ]):
            path = "benchmark/data/scale"
            data = load_infinite_bench(path, dataset)
        else:
            data = load_from_disk(
                f"benchmark/data/longbench/{dataset}"
            )
            data = data.select(range(3))
            

        if len(lines) == len(data):
            print(f'{dataset} completed')
            # for rank in range(args.world_size):
            #     file_path = out_path + f"_{rank}"
            #     os.remove(file_path)
        else:
            print(f'{dataset} incompleted: {len(lines)}/{len(data)}')
