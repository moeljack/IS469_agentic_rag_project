import json
from datasets import load_dataset
from itertools import islice


# Load the dataset in streaming mode
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

# Grab the desired number of samples (e.g., 10 samples)
num_samples = 10
samples = list(islice(dataset, num_samples))

# Write samples to a JSON file
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Successfully exported {len(samples)} samples to dataset.json")

