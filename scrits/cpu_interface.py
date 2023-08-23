import os
import sys

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: PYTHONPATH hacks are never a good idea. clean this up later
sys.path.append(os.path.join(INFERENCE_DIR, '..'))

import cmd
import torch
import argparse
import conversation as convo
import retrieval.wikipedia as wp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
from accelerate import infer_auto_device_map, init_empty_weights



# init
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModelForCausalLM.from_pretrained("medicalai/ClinicalBERT", torch_dtype=torch.bfloat16)

# infer
inputs = tokenizer("<human>: Hello!\n<bot>:", return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
output_str = tokenizer.decode(outputs[0])
print(output_str)