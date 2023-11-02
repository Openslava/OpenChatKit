import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Load data
with open('00_example_gpt_data.json', 'r') as f:
    data = json.load(f)['data']

# Use GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

# Prepare data
inputs = tokenizer([item['qa'] for item in data], return_tensors='pt', padding=True, truncation=True).input_ids

# Define model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
optimizer = AdamW(model.parameters())

# Hyperparameters
epochs = 3  # Define the number of epochs you want. Here, I've set it to 3 as an example.

# Train model
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Save model
model.save_pretrained('./models/gpt_model/')
tokenizer.save_pretrained('./models/gpt_model/')
