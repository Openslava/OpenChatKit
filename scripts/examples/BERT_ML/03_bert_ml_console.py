import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model_path = "./models/bert_multilingual"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def interpret_label(label):
    if label == 0:
        return "Slovak"
    elif label == 1:
        return "English"
    else:
        return "Unknown"

def main():
    while True:
        # Get input from user
        user_input = input("Enter your text (or 'exit' to quit): ")
        
        # Exit loop if user types 'exit'
        if user_input.lower() == 'exit':
            break
        
        # Predict and interpret
        label = predict(user_input, model, tokenizer)
        label_name = interpret_label(label)
        print(f"Predicted Language: {label_name}")

if __name__ == "__main__":
    main()
