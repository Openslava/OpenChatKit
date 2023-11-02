import os
import argparse
import torch
import json

from transformers import AutoModelForCausalLM
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer, LineByLineTextDataset, DataCollatorForLanguageModeling

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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained(model_path)

    # Get the configuration for the LLM model
    # config = AutoConfig(
    #    num_layers=12,
    #    num_heads=8,
    #    vocab_size=len(tokenizer),
    #    hidden_size=768
    #)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    # open existing LLM model for 

    # code to load Dataset in python from file and use tokenizer to train on this dataset 
    # Load the dataset from a file
    def load_dataset(file_path):
        with open(file_path, 'r') as file:
            dataset = file.read()
        return dataset

    # Tokenize the dataset using NLTK tokenizer
    def tokenize_dataset(dataset):
        tokens = word_tokenize(dataset)
        return tokens

    # code that will add tokens generated by tokenizer into LLM model and save it to desired path
    # from transformers import LineByLineTextDataset, LLMTrainingArguments, LLMConfig, LLMForCausalLM, LLMTokenizer, Trainer

    # Define the path of your input text file
    input_file_path = "./data/OIG/files/unified_basic.jsonl"

    # Define the path where you want to save the model
    output_model_path = "./build/trained_model"

    # Initialize the tokenizer
    # tokenizer = LLMTokenizer.from_pretrained('gpt2')

    # Initialize the line-by-line text dataset
    dataset = load_dataset(input_file_path) 
    # LineByLineTextDataset(
    #    tokenizer=tokenizer,
    #    file_path=input_file_path,
    #    block_size=128
    #)

    # Initialize the LLM model 'LLMForCausalLM'
    # model = LLMForCausalLM(config=config)

    device = torch.device('cpu')
    input_model.to(device)

    # Initialize the training arguments
    training_args = TrainingArguments(
        output_dir=output_model_path,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=100,
        save_total_limit=2
    )

    # Initialize the trainer
    trainer = Trainer(
        model=input_model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset
    )

    # Fine-tune the LLM model
    ## trainer.train()

    # Save the tokenizer and model to the desired path
    ## tokenizer.save_pretrained(output_model_path)
    ## model.save_pretrained(output_model_path)

    inputs = tokenizer("<human>: Hello!\n<bot>: yes!", return_tensors='pt').to(input_model.device)
    outputs = input_model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.8)
    output_str = tokenizer.decode(outputs[0])
    print(output_str)

    # outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    output_path    = "./dist/output"
    tokenizer.save_pretrained(output_path)
    outputs.save_pretrained(output_path)
    output_config = outputs.config
    output_config.save_pretrained(output_path)
    
    from torch.utils.data import DataLoader

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)


    # vocabulary = tokenizer.get_vocab().keys()
    # model.resize_token_embeddings(len(tokenizer))


