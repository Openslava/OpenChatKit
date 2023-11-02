#
#
import json

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the JSON file
with open('00_example_data_sk.json', 'r') as file:
    data = json.load(file)

model_cache_dir = "./build/bert/model_cache_dir/"
tokenizer_cache_dir = "./build/bert/model_cache_dir/"

# Load the multilingual BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', cache_dir=model_cache_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir=tokenizer_cache_dir)

# Sample Slovak data for sentiment analysis
data = [
    {"text": "Milujem strojové učenie.", "label": 1}, # "I love machine learning."
    {"text": "Toto je pre mňa príliš ťažké.", "label": 0}  # "This is too difficult for me."
]

# Convert labels and tokenize data
labels = [item['label'] for item in data]
texts = [item['text'] for item in data]
encodings = tokenizer(texts, truncation=True, padding=True)

# Define training args and train the model
training_args = TrainingArguments(per_device_train_batch_size=8, logging_dir='./build/bert/bert_ml_vanila_logs', output_dir='./build/bert/bert_ml_vanila_results')
trainer = Trainer(model=model, args=training_args, train_dataset=encodings, train_labels=labels)
trainer.train()

model.save_pretrained('./build/bert/bert_ml_vanila/')
tokenizer.save_pretrained('./build/bert/bert_ml_vanila/')
