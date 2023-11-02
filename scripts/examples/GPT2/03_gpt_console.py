from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_answer(model, tokenizer, question, max_length=100):
    # Encode question and decode answer
    input_ids = tokenizer.encode(question, return_tensors='pt')
    
    # Generate a response from the model
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output and return the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # Load the trained model and tokenizer
    model_path = './build/essential_oils_gpt_model/'
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Console interaction loop
    print("Ask a question (or type 'exit' to stop):")
    while True:
        user_question = input("> ")
        if user_question.lower() == 'exit':
            break
        
        answer = generate_answer(model, tokenizer, user_question)
        print("Answer:", answer)
