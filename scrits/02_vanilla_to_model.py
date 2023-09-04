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

def create_emtpy_model(model_path):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = AutoModelForCausalLM.from_pretrained(model_path).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=16, ):
    input_path = checkpoint_path

    n_layers = len(model.model.layers)
    assert n_stages * n_layer_per_stage >= len(model.model.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.model.embed_tokens.weight.data[:] = _tmp['embed_tokens.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                ret = model.model.layers[j].load_state_dict(_tmp, strict=False)
                if len(ret.missing_keys):
                    print('The following weight keys are missing:')
                    print(ret.missing_keys)
                if len(ret.unexpected_keys):
                    print('The following weight keys are unexpected:')
                    print(ret.unexpected_keys)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                if i*n_layer_per_stage + j == n_layers:
                    break
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                ret = model.model.layers[i*n_layer_per_stage + j].load_state_dict(_tmp, strict=False)
                if len(ret.missing_keys):
                    print('The following weight keys are missing:')
                    print(ret.missing_keys)
                if len(ret.unexpected_keys):
                    print('The following weight keys are unexpected:')
                    print(ret.unexpected_keys)
            else:
                j += 1

            _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.model.norm.weight.data[:] = _tmp['norm.weight']
            if 'norm.bias' in _tmp:
                model.model.norm.bias.data[:] = _tmp['norm.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                ret = model.model.layers[i*n_layer_per_stage + j].load_state_dict(_tmp, strict=False)
                if len(ret.missing_keys):
                    print('The following weight keys are missing:')
                    print(ret.missing_keys)
                if len(ret.unexpected_keys):
                    print('The following weight keys are unexpected:')
                    print(ret.unexpected_keys)

    return model


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

    vanilla_save_path = "./build/vanilla"

    save_path = "./build/model"
    print('creating empty model...')
    model = create_emtpy_model(vanilla_save_path)
    # if args.fp16:
    #   model = model.half()
    #print('loading model ckpt...')
    #load_decentralized_checkpoint(
    #   model, args.ckpt_path, n_stages=args.n_stages, n_layer_per_stage=args.n_layer_per_stage,
    #)
    #print('loaded model ckpt.')
    
    print(f'saving model  to `{save_path}`')
    model.save_pretrained(save_path)

    print('creating config from model...')
    configuration = model.config
    print(f'saving config  to `{save_path}`')
    configuration.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(vanilla_save_path)
    tokenizer.save_pretrained(save_path)




