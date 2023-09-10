#!/usr/bin/env python

import os
import argparse
import torch
import json
import torch.nn as nn

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
    parser.add_argument('--config-name', type=str, default='./dataset_to_model.json',
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

    from transformers import GPTNeoXConfig, GPTNeoXModel, GPTNeoXTokenizerFast, GPTNeoXForCausalLM

    vanilla_save_path = "./build/vanilla"

    print('creating new vanilla_config...')
    vanilla_config = GPTNeoXConfig()
    vanilla_config.vocab_size = 1000
    vanilla_config.layer_norm_eps = 1e-05
    vanilla_config.intermediate_size = 1000
    vanilla_config.rotary_emb_base = 1000
    vanilla_config.hidden_size = 1000
    vanilla_config.num_attention_heads = 10
    vanilla_config.eos_token_id = 0
    # config_attib = vars(config)
    # json_str = json.dumps(config_attib)
    # print(json_str)
    print(f'saving new vanilla_config to `{vanilla_save_path}`')
    vanilla_config.save_pretrained(vanilla_save_path)

    # attributes = dir(config)
    print('creating new vanilla_model...')
    vanilla_model = GPTNeoXForCausalLM(vanilla_config)
    # load config
    print('creating empty vanilla_model...')
    configuration = vanilla_model.config
    print(f'saving new vanilla_model  to `{vanilla_save_path}`')
    vanilla_model.save_pretrained(vanilla_save_path)

    print(f'saved vanilla_config to `{vanilla_save_path}`')
    configuration.save_pretrained(vanilla_save_path)

    ## https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/tokenizer
    ## * https://huggingface.co/docs/transformers/v4.32.1/en/fast_tokenizers
    ## * https://huggingface.co/docs/tokenizers/quicktour
    #
    # wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    # unzip wikitext-103-raw-v1.zip
    #
    print('create new vanila_tokenizer...')
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    vanila_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    vanila_tokenizer.pre_tokenizer = Whitespace()
    files = [f"/mnt/d/repo/openslava/ai-eng/ai_OpenChatKit/build/data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]    
    vanila_tokenizer.train(files, trainer)
    vanila_tokenizer.save("./build/vanilla/tokenizer-wiki.json")    
    #tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

    print('save new vanila_tokenizer...')
    tokenizer = GPTNeoXTokenizerFast(tokenizer_object=vanila_tokenizer)
    tokenizer.save_pretrained(vanilla_save_path)

# END