import tiktoken
import json
import torch
import argparse
from model import GPT
from tokenization_utils import text_to_token_ids, token_ids_to_text
from generate import generate

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath")

with open('config.json', 'r') as f:
    config = json.load(f)

def main():

    args = parser.parse_args()
    weights_filepath = args.filepath

    # necessary updates for openai weights
    config.update({"context_length": 1024})
    config.update({"qkv_bias": True})

    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT(config)

    checkpoint = torch.load(weights_filepath, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=config["context_length"],
        top_k=50,
        temperature=1.5
    )
    
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == '__main__':
    main()