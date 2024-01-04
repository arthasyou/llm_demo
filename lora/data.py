import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# -------------------------------------------------------------
# 数据处理
# -------------------------------------------------------------

def generate_origin(example):
    r = example + "</s>" 
    return r

def format_zyya(sample):
    r = generate_origin(sample['text'])
    result = tokenizer(r, max_length=1024, padding='max_length')
    input_ids = move_to_end(result["input_ids"], result["input_ids"][0])
    attention_mask = move_to_end(result["attention_mask"], 0)

    result["input_ids"] = input_ids
    result["attention_mask"] = attention_mask

    result["labels"] = result["input_ids"].copy()
    return result

def move_to_end(arr, target):
    count_target = arr.count(target)
    arr = [x for x in arr if x != target]
    arr.extend([target] * count_target)
    return arr

# -------------------------------------------------------------
# 数据处理 end
# -------------------------------------------------------------

# main


tokenizer = AutoTokenizer.from_pretrained(
   "/home/ysx/models/chinese-alpaca-2-7b",
)

# Data

zydata = load_from_disk("/home/ysx/src/AI/llm_demo/data/datasets/zypt")
print(zydata, "\n")
data_token_0 = zydata.map(
    format_zyya,
    remove_columns=['text']
)

print(data_token_0, "\n")
print(tokenizer.decode(data_token_0[0]["input_ids"]), "\n")
