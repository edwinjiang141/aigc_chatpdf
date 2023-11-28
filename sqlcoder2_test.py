import torch
import html2txt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#print(torch.cuda.is_available())
model_name = "defog/sqlcoder2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    load_in_4bit=True,
    device_map="auto",
    use_cache=True,
)