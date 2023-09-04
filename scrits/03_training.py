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

    model_path = "./build/model"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    output_path    = "./dist"
    tokenizer.save_pretrained(output_path)
    outputs.save_pretrained(output_path)
    output_config = outputs.config
    output_config.save_pretrained(output_path)

