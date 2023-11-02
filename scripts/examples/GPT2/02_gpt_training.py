import json


TOKENIZER_CACHE_DIR = './build/tokenizer_cache_dir/'
CACHE_DIR = './build/cache_dir/'
MODEL_DIR = './build/model_dir/'
DATA_DIR = './build/data_dir/'

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium",cache_dir=TOKENIZER_CACHE_DIR)
model = GPT2LMHeadModel.from_pretrained("gpt2-medium",cache_dir=CACHE_DIR)

# Tokenize the data
tokenizer.pad_token = tokenizer.eos_token

# Load data
with open('00_example_gpt2_eo.json', 'r') as file:
    data = json.load(file)

# Concatenate data into text format
text_data = ""
for oil in data:
    for key, value in oil.items():
        text_data += f"{key}: {value}\n"
    text_data += "\n"

# encoded_data = tokenizer.encode(text_data, return_tensors="pt")
encoded_data = tokenizer.encode(text_data, return_tensors='pt', truncation=True, max_length=1024)

with open('./build/00_example_gpt2_eo_encoded.txt', 'w') as f:
    for token_id in encoded_data[0]:
        f.write(str(token_id) + '\n')

with open('./build/00_example_gpt2_eo.txt', 'w') as ftxt:
    ftxt.write(text_data)

# Create Dataset and DataLoader
dataset = TextDataset(
    tokenizer=tokenizer,
    block_size=128,
    overwrite_cache=True,
    file_path='./build/00_example_gpt2_eo.txt'
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./build/results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    report_to=[],  # Disable all integrations including wandb
     logging_dir='./build/logs'    
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

trainer.save_model("./build/essential_oils_gpt_model")
tokenizer.save_pretrained("./build/essential_oils_gpt_model")
