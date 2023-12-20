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
    r = example
    return r

def format_zyya(sample):
    r = generate_origin(sample['text'])
    result = tokenizer(r, max_length=1024, padding='max_length')

    result["labels"] = result["input_ids"].copy()
    return result


# -------------------------------------------------------------
# 数据处理 end
# -------------------------------------------------------------

# main
model = AutoModelForCausalLM.from_pretrained(
    "/home/ysx/models/internlm-chat-7b",
    load_in_4bit=True,
    device_map='auto',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
   "/home/ysx/models/internlm-chat-7b",
   trust_remote_code=True
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


# Data

zydata = load_from_disk("/home/ysx/src/AI/llm_demo/data/datasets/zypt")
data_token_0 = zydata.map(
    format_zyya,
    remove_columns=['text']
)

print(tokenizer.decode(data_token_0[0]["input_ids"]), "\n")

# 

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=data_token_0,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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

