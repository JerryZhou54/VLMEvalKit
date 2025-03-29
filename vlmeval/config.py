from vlmeval.vlm import llama_vision_cot
from functools import partial

llama_series={
    'LLaVA-CoT': partial(llama_vision_cot, model_path='Xkev/Llama-3.2V-11B-cot'),
}

supported_VLM = {}

model_groups = [
    llama_series
]

for grp in model_groups:
    supported_VLM.update(grp)
