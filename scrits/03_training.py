import os
import argparse
import torch
import json

from transformers import AutoModelForCausalLM
from transformers import AutoConfig, AutoModel, AutoTokenizer

from transformers.modeling_utils import no_init_weights

import os

import signal

def handler(signum, frame):
    print('Signal handler called with signal', signum)

signal.signal(signal.SIGABRT, handler)
signal.signal(signal.SIGINT, handler)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--config-name', type=str, default='./build/config.json',
                        help='config-name')
    parser.add_argument('--ckpt-path', type=str, default=None, 
                        help='ckpt-path')
    parser.add_argument('--save-path', type=str, default='./build', 
                        help='save-path')
    parser.add_argument('--n-stages', type=int, default=8, 
                        help='pipeline group size')
    parser.add_argument('--n-layer-per-stage', type=int, default=4, 
                        help='n layers per GPU device')
    parser.add_argument('--fp16', default=False, action='store_true')
    args = parser.parse_args()
    
    # assert args.ckpt_path is not None
    assert args.save_path is not None
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    # * Train with a script * https://huggingface.co/docs/transformers/run_scripts
    # * https://github.com/huggingface/transformers/tree/main 
    # * Processing the data * https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt
    # * Pythia-Chat-Base-7B * https://huggingface.co/togethercomputer/Pythia-Chat-Base-7B/blob/main/README.md

    model_path = "./build/model"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    inputs = tokenizer("<human>: Hello!\n<bot>:", return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
    output_str = tokenizer.decode(outputs[0])
    print(output_str)

    # outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    output_path    = "./dist/output"
    tokenizer.save_pretrained(output_path)
    outputs.save_pretrained(output_path)
    output_config = outputs.config
    output_config.save_pretrained(output_path)

