import torch
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# Ensure CPU usage
device = torch.device("cpu")

# Load the data
with open('data/data.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

# Dataset creation
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = CustomDataset(encodings, labels)

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2).to(device)

# Training setup
training_args = TrainingArguments(
    output_dir='./models/results',
    num_train_epochs=3,
    per_device_train_batch_size=2,  
    logging_dir='./logs',
    do_train=True,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    push_to_hub=False,  # Ensure no pushing to HuggingFace Hub
    report_to="none",  # Reduce logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("./models/bert_multilingual")
tokenizer.save_pretrained("./models/bert_multilingual")
