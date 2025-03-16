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

text = '''
You are a helpful assistant. Below is a long document containing multiple paragraphs about different topics including biology, history, space exploration, artificial intelligence, and philosophy. At the end of the document, a question will be asked, and your task is to answer it based only on the relevant parts of the document.

--- Start of Document ---

[Biology]
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a by-product. Plants also undergo respiration, where they break down sugar molecules to release energy.

[History]
The Roman Empire, at its peak, encompassed most of Europe, parts of the Middle East, and North Africa. Julius Caesar played a significant role in its transition from Republic to Empire. The empire eventually declined due to internal conflict, economic troubles, and invasions by barbarian tribes.

[Space Exploration]
The first successful landing on the Moon occurred in 1969 with NASA's Apollo 11 mission. Neil Armstrong and Buzz Aldrin were the first humans to walk on the lunar surface. Since then, numerous missions have explored Mars, the outer planets, and distant galaxies using telescopes and space probes like Voyager and James Webb.

[Artificial Intelligence]
Artificial intelligence (AI) is the simulation of human intelligence in machines. AI systems are capable of learning from data, recognizing patterns, and making decisions. Machine learning, deep learning, and reinforcement learning are key areas of modern AI. GPT-based language models are powerful examples of generative AI systems.

[Philosophy]
Socrates, Plato, and Aristotle are foundational figures in Western philosophy. Socratic questioning promotes critical thinking. Plato's theory of forms and Aristotle's logic have influenced thought for centuries. Eastern philosophy also offers rich traditions, such as Confucian ethics and Buddhist teachings on impermanence and mindfulness.

--- End of Document ---

Now answer this question:
Which two astronauts walked on the Moon during the Apollo 11 mission, and what year did this event occur?

'''

encoded_text = tokenizer.encode(text)
input_ids = torch.tensor(encoded_text).unsqueeze(0).to("cuda:0")

output = model.generate(input_ids, max_length=200)
print(output)
model.clear()