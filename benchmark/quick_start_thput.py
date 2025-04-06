import time
import torch
import csv
from qllm.models import LlamaForCausalLM
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from qllm.utils import patch_hf, GreedySearch, patch_model_center

# 加载配置
conf = OmegaConf.load("./config/llama3-qllm.yaml")
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, add_bos_token=True, add_eos_token=False
)

model = patch_hf(model, "qllm", conf.model)
model = GreedySearch(model, tokenizer)

# 测试用 prompt
text = '''
You are a helpful assistant. Below is a long document containing multiple paragraphs about different topics including biology, history, space exploration, artificial intelligence, and philosophy.

[Biology] Photosynthesis is the process by which green plants...

[History] The Roman Empire, at its peak...

[Space Exploration] The first successful landing on the Moon...

[Artificial Intelligence] Artificial intelligence (AI)...

[Philosophy] Socrates, Plato, and Aristotle...

Now answer this question:
Which two astronauts walked on the Moon during the Apollo 11 mission, and what year did this event occur?
'''

input_ids = tokenizer.encode(text)
input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda:0")

# 测试不同 max_new_tokens
test_tokens_list = [512, 1024, 2048]
output_file = "thput_result.csv"

# 写入 CSV header
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["max_new_tokens", "tokens_generated", "time_cost(s)", "throughput(tokens/s)"])

# Benchmark 测试循环
for max_new_tokens in test_tokens_list:
    torch.cuda.synchronize()
    start = time.time()

    # 推荐用 max_new_tokens 保证追加 token 数量
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)

    torch.cuda.synchronize()
    end = time.time()

    tokens_generated = len(output[0]) - input_ids.shape[1]
    tokens_generated = max(tokens_generated, 0)  # 防止负数

    time_cost = end - start
    throughput = tokens_generated / time_cost if time_cost > 0 else 0

    print(f"[max_new_tokens={max_new_tokens}] Generate {tokens_generated} tokens in {time_cost:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/s")

    with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([max_new_tokens, tokens_generated, time_cost, throughput])

model.clear()
