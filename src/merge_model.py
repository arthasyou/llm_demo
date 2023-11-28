from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

peft_model_id = "/Users/you/src/llm_demo/outputs/zylora"
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
# merged_model.config.save_pretrained("../outputs/zysft")
tokenizer.save_pretrained("../outputs/zysft")
merged_model.save_pretrained("../outputs/zysft")