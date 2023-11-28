from datasets import Dataset

dataset = Dataset.from_json("../data/json/alpaca_data_cleaned_archive.json")
dataset.save_to_disk("../data/datasets/zysft")