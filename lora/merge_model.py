from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

peft_model_id = "/Users/you/zypt_llama_13B"

config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="mps",
)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
)

model = PeftModel.from_pretrained(inference_model, peft_model_id, device_map="mps")

merged_model = model.merge_and_unload()
merged_model.config.do_sample = True
merged_model.config.use_cache = True

merged_model.save_pretrained("../outputs/zypt_13B")
merged_model.config.save_pretrained("../outputs/zypt_13B")
tokenizer.save_pretrained("../outputs/zypt_13B")