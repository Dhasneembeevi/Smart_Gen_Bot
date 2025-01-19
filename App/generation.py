import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from functools import lru_cache
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@lru_cache(maxsize=100)
def get_cached_response(query):
    return None

def generate_openai_response(query):
    """Generate a response using OpenAI API."""
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo", 
            prompt=query,
            max_tokens=150,  # Limit output length to avoid hitting token limits
            n=1,
            stop=None,
            temperature=0.7
        )
        
        generated_text = response.choices[0].text.strip()
        return generated_text

    except openai.error.RateLimitError:
        print("Rate limit exceeded with OpenAI, falling back to GPT-2...")
        return None
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request to OpenAI: {e}")
        return "Sorry, there was an error with your OpenAI request."
    except Exception as e:
        print(f"An error occurred with OpenAI: {e}")
        return "Sorry, something went wrong with OpenAI."

def generate_gpt2_response(query):
    """Generate a response using the local GPT-2 model."""
    inputs = tokenizer.encode(query, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_response(query, preprocessed_data):
    cached_response = get_cached_response(query)
    if cached_response:
        print("Returning cached response.")
        return cached_response
    
    openai_response = generate_openai_response(query)
    if openai_response:
        get_cached_response.cache_set(query, openai_response)
        return openai_response
    
    print("Falling back to GPT-2 response.")
    gpt2_response = generate_gpt2_response(query)
    return gpt2_response

def preprocess_data(data):
    return data

def main():
    query = "What is Machine Learning?"
    preprocessed_data = preprocess_data("Some example preprocessed data")

    print(generate_response(query,preprocessed_data))

if __name__ == "__main__":
    main()
