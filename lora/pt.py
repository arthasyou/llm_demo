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

def generate_prompt(example):
    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}<s>"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}<s>"
    )

def format_alpaca_data(sample):
    r = generate_prompt(sample)
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

# main
model = AutoModelForCausalLM.from_pretrained(
    "/home/ysx/models/chinese-alpaca-2-7b",
    load_in_4bit=True,
    # load_in_8bit=True,
    device_map='auto',
)



#Freezing the original weights
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

# model.lm_head = CastOutputToFloat(model.lm_head)

# print(model)

# Setting up the LoRa Adapters
config = LoraConfig(
    r=4, #attention heads
    lora_alpha=8, #alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"], #if you know the
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)




# 

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=data_token_0,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=3,
        max_steps=3000,
        save_steps=200,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir='../outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

