# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm
import os.path as osp
import os
import shutil
import argparse
from omegaconf import OmegaConf
from qllm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer
from qllm.utils.extract_question import extract_question_id
import time 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None, help="number of samples to run per dataset")
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.output_dir_path = args.output_dir_path
    if not osp.exists(conf.output_dir_path):
        os.makedirs(conf.output_dir_path)
    shutil.copy(args.config_path, conf.output_dir_path)
    conf.model.model_center = args.model_center
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    conf.num_samples = args.num_samples
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    return conf


def get_model_and_tokenizer(config, conv_type, TOKEN=None):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, token=TOKEN)
    if config.model_center:
        import bmtrain as bmt
        bmt.init_distributed(seed=233)
        from model_center.model import Llama, LlamaConfig
        model_config = LlamaConfig.from_pretrained(config.path, token=TOKEN)
        model_config.dtype = torch.bfloat16
        model = Llama(model_config)
        bmt.load(model, os.path.join(config.path, "pytorch_model.pt"), strict=False)
        model = patch_model_center(model, config.type, **config)
    else:
        from qllm.models import LlamaForCausalLM, MistralForCausalLM
        if conv_type == 'mistral-inst':
            model = MistralForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, token=TOKEN, trust_remote_code=True, device_map="cuda")
        elif conv_type == 'llama-3-inst':
            model = LlamaForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, token=TOKEN, trust_remote_code=True, device_map="cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, token=TOKEN, trust_remote_code=True, device_map="cuda")
        model = patch_hf(model, config.type, **config)
    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()

    if model_name == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt

def load_infinite_bench(path, data_name) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]
    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0], 
                    "OPTION_B": eg["options"][1], 
                    "OPTION_C": eg["options"][2], 
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name.startswith("kv_retrieval"):
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        if 'answers' not in eg:
            ans = get_answer(eg)
            instance["answers"] = ans if isinstance(ans, list) else [ans]
        else:
            instance["answers"] = eg['answers']
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None
        
        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret


def post_process(pred, model_name):
    if model_name == "qwen":
        return pred.split("<|im_end|>")[0]
    return pred

def get_pred(
    model, tokenizer, data, max_length,
    max_gen, prompt_format, dataset, model_name, 
    gen_chunk_size = None, truncation: str = None, 
    rank: int = None, world_size: int = None,
    verbose: bool = False, out_path: str = None,
    model_type: str = None,
):
    preds = []
    data = list(data)

    if world_size is not None:
        data = data[rank::world_size]

    searcher = GreedySearch(model, tokenizer)
    cur = 0
    total = len(data)

    start_id = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            past_data = [l.strip() for l in f.readlines()]
        past_data = set(past_data)
        start_id = len(past_data)
        # with open(out_path, "w+") as f:
        #     f.write('\n'.join(past_data))
    
    total_token_count = 0
    total_run_time = 0.0

    for json_obj in tqdm(data[start_id:]):
        #print(f"[QLLM pred] Predicting sample: {json_obj}")
        prompt = prompt_format.format(**json_obj)

        extra_end_token_ids = []
        if model_name == "llama-3-inst":
            extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])

        if model_name == "qwen":
            extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

        if dataset == "samsum":
            extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])


        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

            if model_name.strip().lower() in ['mistral-inst']:
                add_special_tokens = False
            else:
                add_special_tokens = True
        
        else:
            add_special_tokens = True

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]

        if truncation is None:
            if len(tokenized_prompt) > max_length - max_gen:
                if verbose:
                    print(f"Length {len(tokenized_prompt)}. Skipped.")
                continue

        else:
            if truncation == "suffix":
                length = len(tokenized_prompt)
                if length > max_length - max_gen:
                    if verbose:
                        print("over length")
                    init_token_num = 128
                    prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            else:
                raise NotImplementedError
    
        kwargs = {}
        if model_type in ['qllm']:
            kwargs["question_ids"] = extract_question_id(
                dataset, tokenizer, tokenized_prompt, json_obj)
            
        start_time = time.time()
        if dataset == "samsum":
            output = searcher.generate(
                input_ids = tokenized_prompt,
                max_length=max_gen,
                extra_end_token_ids=[
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
                chunk_size=gen_chunk_size,
                **kwargs,
            )
        else:
            output = searcher.generate(
                input_ids = tokenized_prompt,
                max_length=max_gen,
                chunk_size=gen_chunk_size,
                **kwargs,
            )
        run_time = time.time() - start_time
        total_run_time += run_time
        total_token_count += len(tokenized_prompt) + len(output[0])
        result = post_process(output[0], model_name)
        pred = {
            "pred": result, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "token_length": len(tokenized_prompt) + max_gen, 'time': run_time,
        }
        searcher.clear()
        cur += 1
        if verbose:
            print(f"----------{cur}/{total}----------")
            print("Question:", prompt[-100:])
            print("Pred:", result)
            print("Answer:", json_obj["answers"])
            print("")

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')
        # import pdb;pdb.set_trace()
    
    throughput = total_token_count / total_run_time if total_run_time > 0 else 0
    print(f"[Throughput] total_tokens={total_token_count}, total_time={total_run_time:.2f}s, throughput={throughput:.2f} tokens/s")
    return preds


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    model, tokenizer = get_model_and_tokenizer(args.model, args.conv_type)
    output_dir_path = args.output_dir_path
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    datasets = args.datasets

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    multiprocessing = args.world_size is not None and args.world_size > 1
    if multiprocessing:
        assert args.rank in list(range(args.world_size))

    # predict on each dataset
    for dataset in datasets:
        dname = dataset
        if dataset in set([
            "kv_retrieval", "passkey", "number_string", "code_run", "code_debug", "longdialogue_qa_eng", "longbook_qa_eng", "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_chn", "math_find", "math_calc"
        ]):
            path = "benchmark/data/infinite-bench"
            data = load_infinite_bench(path, dname)

        elif dataset in set([
            "kv_retrieval_32k", "kv_retrieval_64k", "kv_retrieval_128k", "kv_retrieval_256k", "kv_retrieval_512k", "kv_retrieval_768k", "kv_retrieval_1024k", 
        ]):
            path = "benchmark/data/scale"
            data = load_infinite_bench(path, dname)

        elif dataset.startswith('custom'):
            data = load_infinite_bench("benchmark/data/custom/", dataset)

        else:
            data = load_from_disk(
                f"benchmark/data/longbench/{dataset}"
            )
        if args.num_samples is not None:
            data = data.select(range(args.num_samples))


        out_path = os.path.join(
            output_dir_path,
            f"{dname}.jsonl"
        )
        if os.path.exists(out_path):
            with open(out_path) as f:
                complete_l = len(f.readlines())
            if complete_l == len(data):
                print(f'{dname} completed')
                continue
        
        if multiprocessing:
            out_path = out_path + f"_{args.rank}"

        print(f"Pred {dname}")
        if dataset in set([
            "kv_retrieval_32k", "kv_retrieval_64k", "kv_retrieval_128k", "kv_retrieval_256k", "kv_retrieval_512k", "kv_retrieval_768k", "kv_retrieval_1024k", 
        ]):
            prompt_format = dataset2prompt['kv_retrieval']
            max_gen = dataset2maxlen['kv_retrieval']
        elif dataset.startswith('custom'):
            if dataset.startswith('custom_book'):
                prompt_format = dataset2prompt['custom_book']
            elif dataset.startswith('custom_paper'):
                prompt_format = dataset2prompt['custom_paper']
            else:
                prompt_format = dataset2prompt[dataset]
            max_gen = 2048
        else:
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
        
        get_pred(
            model, tokenizer, data, 
            args.max_len, max_gen, 
            prompt_format, dataset, 
            args.conv_type, 
            args.chunk_size, args.truncation,
            args.rank, args.world_size,
            args.verbose, out_path,
            args.model.type,
        )
        peak_memory = torch.cuda.max_memory_allocated(device='cuda')
        print(f"[Memory] Peak Memory Allocated: {peak_memory / 1024 / 1024:.2f} MB")

