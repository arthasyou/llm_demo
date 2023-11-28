from datasets import Dataset

dataset = Dataset.from_json("../data/json/alpaca.json")
dataset.save_to_disk("../data/datasets/zysft")