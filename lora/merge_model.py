from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

peft_model_id = "/Users/you/zypt_llama_7B"

inference_model = AutoModelForCausalLM.from_pretrained(
    "/Users/you/models/chinese-alpaca-2-7b",
    device_map="mps",
)

tokenizer = AutoTokenizer.from_pretrained(
    "/Users/you/models/chinese-alpaca-2-7b",
)

model = PeftModel.from_pretrained(inference_model, peft_model_id, device_map="mps")

merged_model = model.merge_and_unload()
merged_model.config.do_sample = True
merged_model.config.use_cache = True

merged_model.save_pretrained("../outputs/zypt_7B")
merged_model.config.save_pretrained("../outputs/zypt_7B")
tokenizer.save_pretrained("../outputs/zypt_7B")