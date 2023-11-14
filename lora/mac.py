import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk, concatenate_datasets

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

def generate_prompt(example):
    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )

def replace_qa(text):
    if text.startswith("患者"):
        result = "### User:\n" + text[3:]
        return result
    else:
        result = "### Assist:\n" + text[3:]
        return result
    
def generate_zy_prompt(example):
    f = example[0]
    s = example[1]
    f = replace_qa(f)
    s = replace_qa(s)
    return f"{f}\n{s}"

def generate_origin(example):
    r = "\n".join(example)
    return r

def format_zyya(sample):
    r = generate_origin(sample['text'])
    result = tokenizer(r, max_length=1024, padding='max_length')
    result["labels"] = result["input_ids"].copy()
    return result

def format_chat(sample):
    r = generate_zy_prompt(sample['text'])
    result = tokenizer(r, max_length=1024, padding='max_length')
    result["labels"] = result["input_ids"].copy()
    return result

def format_alpaca_data(sample):
    r = generate_prompt(sample)
    result = tokenizer(r, max_length=1024, padding='max_length')
    result["labels"] = result["input_ids"].copy()
    return result

# main
model = AutoModelForCausalLM.from_pretrained(
    "/Users/you/tmp/internlm-chat-7b",
    device_map='mps',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
   "/Users/you/tmp/internlm-chat-7b",
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
zydata = load_from_disk("/Users/you/src/llm_demo/data/datasets/zyya")
mapped_dataset = zydata.map(
    format_zyya,
    remove_columns=['text']
)

chat_data = load_from_disk("/Users/you/src/llm_demo/data/datasets/zy_chat")
chat_dataset = chat_data.map(
    format_chat,
    remove_columns=['text']
)

junk_data = load_dataset('json', data_files='/Users/you/src/llm_demo/data/json/alpaca_data_cleaned_archive.json')
junk_dataset = junk_data.map(
    format_alpaca_data,
    remove_columns=['instruction', 'output', 'input']
)

combined_dataset = concatenate_datasets([mapped_dataset, chat_dataset, junk_dataset['train']])

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=combined_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
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