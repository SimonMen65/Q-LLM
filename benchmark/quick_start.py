import torch
from qllm.models import LlamaForCausalLM
from transformers import AutoTokenizer
import transformers

from omegaconf import OmegaConf
from qllm.utils import patch_hf, GreedySearch, patch_model_center

conf = OmegaConf.load("./config/tinyLlama-1.1B.yaml")
model_path = "mistralai/Mistral-7B-Instruct-v0.2"
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


model = LlamaForCausalLM.from_pretrained(
	model_path,
	torch_dtype=torch.bfloat16,
	trust_remote_code=True
	).to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_bos_token=True, add_eos_token=False)

model = patch_hf(model, "qllm", conf.model)
model = GreedySearch(model, tokenizer)

text = "Answer the following question:\nWhat is the capital of France?"

encoded_text = tokenizer.encode(text)
input_ids = torch.tensor(encoded_text).unsqueeze(0).to("cuda:0")

output_ids = model.generate(
    input_ids=input_ids,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("=== MODEL OUTPUT ===")
print(output_text)
model.clear()