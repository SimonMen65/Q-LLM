# save_narrativeqa_dataset.py
from datasets import Dataset
import os

data = [
    {
        "id": "narrativeqa-test-1",
        "question": "Who wrote the Harry Potter series?",
        "context": "J.K. Rowling is a British author best known for writing the Harry Potter books.",
        "answer": "J.K. Rowling",
    }
]

ds = Dataset.from_list(data)

output_path = "./benchmark/data/longbench/narrativeqa"
os.makedirs(output_path, exist_ok=True)
ds.save_to_disk(output_path)
