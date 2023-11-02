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

model.lm_head = CastOutputToFloat(model.lm_head)

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
dataset = load_from_disk("/home/ysx/src/AI/llm_demo/data/datasets/lora")

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=3,
        # max_steps=30000,
        save_steps=200,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=10,
        output_dir='../outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()