from transformers import BertTokenizer, BertForQuestionAnswering

# Load pre-trained mBERT model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def get_answer(question, context, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits) 
    answer_end = torch.argmax(outputs.end_logits) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

def main():
    # Load a multilingual context or take user input
    context = input("Enter the context: ")
    
    print("Welcome to the Q&A Console! Ask a question based on the provided context.")
    
    while True:
        user_question = input("\nAsk your question (or 'exit' to quit): ")
        
        if user_question.lower() == 'exit':
            break
        
        answer = get_answer(user_question, context, model, tokenizer)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
