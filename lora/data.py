import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk

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

def create_prompt(content, question, answer):
    # qst = question[0]
    # ans = answer[0]
    prompt_template = f"### Content\n{content}\n\n### Input\n{question}\n\n### Output\n{answer}</s>"
    return prompt_template

def format_data(samples):
    result = tokenizer(samples['text'], max_length=1024, padding='max_length')
    result["labels"] = result["input_ids"].copy()
    return result


# main


tokenizer = AutoTokenizer.from_pretrained(
   "/home/ysx/models/internlm-chat-7b",
   trust_remote_code=True
)






# Data
dataset = load_from_disk("/home/ysx/src/AI/llm_demo/data/datasets/zyqa")
print(dataset)

# newdata = dataset.select(range(100))
mapped_dataset = dataset.map(
    format_data,
    remove_columns=['text'],
    batched=True
)

print(mapped_dataset)

